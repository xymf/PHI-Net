#!/usr/bin/env python
 

import argparse
import logging
import os
import random
import shutil
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fft as fft
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_full_enhance

# ----------------- 参数 -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r"D:\xxu\dataset\ultrasound\CAMUS4_ActiveLearning_Top135", help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CAMUS/UCSP_Improve_GVA_AFP', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=6,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=135,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--max_delta', type=float,
                    default=0.2, help='consistency_rampup')
parser.add_argument('--active_scale', type=float,
                    default=1, help='consistency_rampup')
args = parser.parse_args()


# ----------------- 工具 -----------------
def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# ----------------- 改进模块 1: AdaptiveFrequencyPerturbation -----------------
class AdaptiveFrequencyPerturbation(nn.Module):
    def __init__(self, in_channels, height, width,
                 max_delta=0.1, min_delta=0.01, rampup_steps=30000):
        super().__init__()
        self.max_delta = max_delta
        self.min_delta = min_delta
        self.rampup_steps = rampup_steps
        self.height = height
        self.width = width
        self.global_scale = nn.Parameter(torch.tensor(1.0))

        # 低通滤波器：保护低频
        self.register_buffer('low_pass_filter', self._create_low_pass_filter(height, width))
        # 频率掩码：强调高频
        self.register_buffer('freq_mask', self._create_freq_mask(height, width))

        # 频谱注意力（基于不确定性图）
        self.spectral_attention = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _create_low_pass_filter(self, h, w):
        y = torch.arange(h, dtype=torch.float32) - h // 2
        x = torch.arange(w, dtype=torch.float32) - w // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        d = torch.sqrt(xx ** 2 + yy ** 2)
        cutoff = min(h, w) * 0.2
        filt = torch.exp(-(d ** 2) / (2 * (cutoff ** 2)))
        return filt.unsqueeze(0)

    def _create_freq_mask(self, h, w):
        y = torch.arange(h, dtype=torch.float32) - h // 2
        x = torch.arange(w, dtype=torch.float32) - w // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        d = torch.sqrt(xx ** 2 + yy ** 2)
        max_d = torch.sqrt(torch.tensor(float((h // 2) ** 2 + (w // 2) ** 2)))
        norm_d = d / (max_d + 1e-8)
        mask = torch.clamp(0.2 + 0.8 * norm_d, 0, 1)
        return mask.unsqueeze(0)

    def _get_rampup_factor(self, step):
        step = min(step, self.rampup_steps)
        ratio = step / self.rampup_steps
        return self.min_delta + (self.max_delta - self.min_delta) * ratio

    def forward(self, x, step, uncertainty=None):
        B, C, H, W = x.shape
        device = x.device
        current_delta = self._get_rampup_factor(step)

        results = []

        if (H, W) == (self.height, self.width):
            low_pass = self.low_pass_filter.to(device)
            freq_mask = self.freq_mask.to(device)
        else:
            low_pass = self._create_low_pass_filter(H, W).to(device)
            freq_mask = self._create_freq_mask(H, W).to(device)

        for b in range(B):
            img = x[b:b + 1]

            # FFT
            fft_img = torch.fft.fft2(img, dim=(-2, -1))
            mag = torch.abs(fft_img)
            pha = torch.angle(fft_img)

            # 频谱注意力
            if uncertainty is not None and uncertainty.numel() > 0:
                u = uncertainty[b:b + 1]
                att = self.spectral_attention(u)
            else:
                att = torch.ones(1, 1, H, W, device=device, dtype=mag.dtype)

            att = att.expand(-1, C, -1, -1)
            fm = freq_mask.unsqueeze(1).expand(-1, C, -1, -1)
            lp = low_pass.unsqueeze(1).expand(-1, C, -1, -1)

            # 幅值扰动
            strength = current_delta * fm * (1.0 - lp) * att * self.global_scale
            mag_perturbed = mag * (1.0 + strength)

            # --- [关键改进] 移除相位随机噪声 ---
            # 相位包含了图像的结构/边缘信息，随机扰动相位会导致解剖结构错乱
            # 保持原始相位不变，只改变幅值（风格/纹理）
            pha_perturbed = pha

            try:
                fft_perturbed = torch.polar(mag_perturbed, pha_perturbed)
                img_perturbed = torch.fft.ifft2(fft_perturbed, dim=(-2, -1)).real
                results.append(img_perturbed)
            except Exception as e:
                results.append(img)

        if len(results) == 0:
            return x
        return torch.cat(results, dim=0)


# ----------------- 改进模块 2: GVA 网络 -----------------
class GVA2(nn.Module):
    """
    GVA2: 保持原结构，优化初始化，配合 Fidelity Loss 使用
    """

    def __init__(self, layers: int = 3, channels: int = 64, reduction: int = 4,
                 in_channels: int = 1, out_channels: int = 1, checkpoint: bool = False):
        super().__init__()
        self.layers = layers
        self.checkpoint = checkpoint

        self.in_conv = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.in_conv.weight, mode='fan_out', nonlinearity='relu')

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(Bottleneck(channels, reduction))

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        # 初始化为0，保证初始状态近似恒等映射，减少训练初期的震荡
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        fea = self.in_conv(x)
        g = self.gate(fea)

        for blk in self.blocks:
            if self.checkpoint and self.training:
                fea = torch.utils.checkpoint.checkpoint(blk, fea)
            else:
                fea = blk(fea)

        # [修改处] ------------------------------------------------
        # 强制限制 scale 的最小值，防止它训练成 0
        # 使用 softplus 或 clamp 都可以，这里用 clamp 最直接
        # 注意：这里我们取 abs 确保方向一致，并且加一个 min 限制
        active_scale = torch.clamp(torch.abs(self.scale), min=0.05, max=args.active_scale)

        delta = self.out_conv(fea) * active_scale
        # --------------------------------------------------------

        out = torch.clamp(identity + delta, 0.0, 1.0)
        enhanced = g * out + (1 - g) * identity
        return enhanced

    # 部署接口
    @torch.no_grad()
    def enhance(self, img: torch.Tensor) -> torch.Tensor:
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = img.to(next(self.parameters()).device)
        self.eval()
        out = self.forward(img)
        out = (out * 255.0).round().clamp(0, 255).to(torch.uint8)
        return out


class Bottleneck(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        mid = channels // reduction
        self.preact = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        nn.init.zeros_(self.conv[-1].weight)

    def forward(self, x):
        residual = x
        x = self.preact(x)
        x = self.conv(x)
        return x + residual


def soft_dice_loss(inputs, targets, exclude_background=False):
    smooth = 1e-5
    inputs = F.softmax(inputs, dim=1)
    intersection = torch.sum(inputs * targets, dim=(2, 3))
    union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    if exclude_background:
        dice = dice[:, 1:]
    mean_dice = dice.mean()
    return 1 - mean_dice


# ----------------- 训练主函数 -----------------
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    warmup_iters = 3000

    # ---------- 初始化 ----------
    gva_enhancer = GVA2(layers=3, channels=64, in_channels=1, out_channels=1).cuda()

    # 冻结 GVA2 (Warmup)
    for p in gva_enhancer.parameters():
        p.requires_grad = False

    fft_perturb = AdaptiveFrequencyPerturbation(
        in_channels=1, height=args.patch_size[0], width=args.patch_size[1],
        max_delta=args.max_delta, min_delta=0.01, rampup_steps=max_iterations
    ).cuda()

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model().cuda()
    ema_model = create_model(ema=True).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train",
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = args.labeled_num
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(
        list(model.parameters()),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    cls_weight_ce = torch.tensor([0.10, 0.30, 0.30, 0.30]).cuda()
    cls_weight_dice = torch.tensor([0.10, 0.30, 0.30, 0.30]).cuda()
    ce_loss = nn.CrossEntropyLoss(weight=cls_weight_ce)
    dice_loss = losses.DiceLoss_weight(num_classes, weight=cls_weight_dice)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # ===== 1. GVA2 增强 & 保真度 Loss [重要改进] =====
            # 备份原图，用于计算保真度 Loss
            volume_batch_original = volume_batch.clone().detach()

            # 增强
            volume_batch = gva_enhancer(volume_batch)

            # Fidelity Loss: 强制增强后的图像与原图保持结构一致 (L1 Loss)
            loss_fidelity = F.l1_loss(volume_batch, volume_batch_original)

            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            # ===== 2. 教师分支：FFT 扰动 =====
            if unlabeled_volume_batch.shape[0] > 0:
                with torch.no_grad():
                    # 教师输入 = GVA2 输出 + FFT 扰动
                    ema_inputs = fft_perturb(unlabeled_volume_batch, iter_num)
                    ema_output = ema_model(ema_inputs)
            else:
                ema_inputs = unlabeled_volume_batch
                ema_output = None

            # ===== 3. 学生前向 =====
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # ===== 4. MC-Dropout 扰动 (保持原逻辑) ======
            T = 8
            if unlabeled_volume_batch.shape[0] > 0:
                _, _, w, h = unlabeled_volume_batch.shape
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
                for i in range(T // 2):
                    with torch.no_grad():
                        # 使用改进后的 fft_perturb (无相位噪声)
                        ema_inputs_mc = fft_perturb(volume_batch_r, iter_num)
                        preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs_mc)
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, num_classes, w, h).mean(dim=0)
                max_prob = preds.max(dim=1).values
                # 保持原有的不确定性计算逻辑
                uncertainty = 1.0 - max_prob.unsqueeze(1)
            else:
                uncertainty = torch.zeros(0, 1, 0, 0).cuda()

            # ===== 5. 损失计算 (保持原逻辑 + Fidelity) ======
            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.3 * loss_ce + 0.7 * loss_dice

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            if unlabeled_volume_batch.shape[0] > 0 and ema_output is not None:
                # 保持原来的损失逻辑
                stu_prob = torch.softmax(outputs[args.labeled_bs:], dim=1)
                tea_prob = torch.softmax(ema_output, dim=1)

                consistency_dist_spatial = F.smooth_l1_loss(stu_prob, tea_prob, reduction='none').mean(dim=1,
                                                                                                       keepdim=True)

                # 频域一致性逻辑 (保持原样)
                stu_fft = torch.fft.fft2(stu_prob, dim=(-2, -1), norm='ortho')
                tea_fft = torch.fft.fft2(tea_prob, dim=(-2, -1), norm='ortho')
                stu_log_mag = torch.log(torch.abs(stu_fft) + 1e-8)
                tea_log_mag = torch.log(torch.abs(tea_fft) + 1e-8)
                stu_shifted = torch.fft.fftshift(stu_log_mag, dim=(-2, -1))
                tea_shifted = torch.fft.fftshift(tea_log_mag, dim=(-2, -1))

                # 原来的 Mask 逻辑
                B_u, C_u, H_u, W_u = stu_shifted.shape
                radius = min(H_u, W_u) * 0.15
                y = torch.arange(H_u, device=stu_shifted.device) - H_u // 2
                x = torch.arange(W_u, device=stu_shifted.device) - W_u // 2
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                dist_map = torch.sqrt(xx ** 2 + yy ** 2)
                mask_high = torch.sigmoid((dist_map - radius) * 0.5).view(1, 1, H_u, W_u)
                mask_low = 1.0 - mask_high
                freq_diff_sq = (stu_shifted - tea_shifted) ** 2
                loss_freq_map = (freq_diff_sq * mask_low * 0.5) + (freq_diff_sq * mask_high * 1.0)
                freq_loss_scalar = loss_freq_map.mean(dim=(-2, -1), keepdim=True).mean(dim=1, keepdim=True)

                freq_lambda = 0.1 * ramps.sigmoid_rampup(iter_num // 150, args.consistency_rampup)
                dist_map_combined = consistency_dist_spatial + freq_lambda * freq_loss_scalar
                loss_cons_dice = soft_dice_loss(outputs[args.labeled_bs:], tea_prob.detach(), exclude_background=False)

                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                mask_binary = (uncertainty < threshold).float()
                weight_soft = 2.0 * torch.exp(-5 * uncertainty)
                valid_pixel_weights = mask_binary * weight_soft

                weighted_consistency_loss = (dist_map_combined * valid_pixel_weights).sum() / (
                        valid_pixel_weights.sum() + 1e-8)
                consistency_loss = weighted_consistency_loss + loss_cons_dice
            else:
                consistency_loss = torch.tensor(0.0).cuda()

            # [重要] Total Loss 加入 Fidelity Loss
            loss = 1.0 * consistency_loss + supervised_loss + 0.1 * loss_fidelity

            # ===== 6. 反向 & EMA 更新 =====
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            # GVA2 解冻
            if iter_num == warmup_iters:
                logging.info(f"Warm-up 结束，解冻 GVA2，开始端到端训练（iter={iter_num}）")
                for p in gva_enhancer.parameters():
                    p.requires_grad = True
                optimizer.add_param_group({
                    'params': gva_enhancer.parameters(),
                    'lr': lr_,
                    'momentum': 0.9,
                    'weight_decay': 0.0001,
                })

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/loss_fidelity', loss_fidelity, iter_num)

            logging.info(
                'iteration %d : loss : %f, cons: %f, fidelity: %f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_fidelity.item()))

            # ===== 7. 可视化 & 验证 =====
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Enhanced_Image', image, iter_num)
                writer.add_image('train/Original_Image', volume_batch_original[1, 0:1, :, :], iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch_val in enumerate(valloader):
                    # 注意：这里需要传入 gva_enhancer
                    metric_i = test_single_volume_full_enhance(
                        sampled_batch_val["image"], sampled_batch_val["label"], model,
                        classes=num_classes, enhancer=gva_enhancer)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    # 保存 GVA2
                    torch.save(gva_enhancer.state_dict(), os.path.join(snapshot_path, 'best_gva.pth'))

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


# ----------------- main -----------------
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    snapshot_path = "../compare/UCSP_Refined_ord/{}_{}_labeled_/{}".format(
        args.exp, args.labeled_num, args.model)
    # snapshot_path = "../compare/UCSP_Refined_Ordinary/{}_{}_labeled_max_delta_{}_active_scale_{}/{}".format(
    #     args.exp, args.labeled_num, args.model)
    # snapshot_path = "../compare/UCSP_Ablation_Moudle/parameter_Ablation/{}_{}_labeled_max_delta_{}_active_scale_{}/{}".format(
    #     args.exp, args.labeled_num, args.max_delta, args.active_scale,args.model, )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
