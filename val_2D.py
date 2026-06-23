import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

import torch.nn as nn


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        slice = torch.from_numpy(slice) / 255
        input_d = slice.unsqueeze(0).unsqueeze(0).cuda()
        model.eval()
        with torch.no_grad():
            output = model(input_d)
            if len(output) > 1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
class Bottleneck(nn.Module):
    """Pre-Activation Bottleneck: BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)-Conv(1×1)"""
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
        # 初始化最后一层 Conv 权重为 0，保证初始恒等映射
        nn.init.zeros_(self.conv[-1].weight)

    def forward(self, x):
        residual = x
        x = self.preact(x)
        x = self.conv(x)
        return x + residual
class GVA2(nn.Module):
    """
    GVA2: Learnable Gated Video/Image Enhancement
    1. 可学习亮度门控（像素级掩码）
    2. Batch-wise 并行，无 for-loop
    3. Pre-Activation Bottleneck 残差
    """

    def __init__(self,
                 layers: int = 3,
                 channels: int = 64,
                 reduction: int = 4,          # bottleneck 压缩倍率
                 in_channels: int = 1,
                 out_channels: int = 1,
                 checkpoint: bool = False):
        super().__init__()
        self.layers = layers
        self.checkpoint = checkpoint

        # 输入投影
        self.in_conv = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.in_conv.weight, mode='fan_out', nonlinearity='relu')

        # 可学习门控：全局统计 → 1×1 → Sigmoid
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(Bottleneck(channels, reduction))

        # 输出头
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        # 对 delta 做缩放：初始化为 0，更接近恒等映射
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        fea = self.in_conv(x)

        # 门控掩码（B,1,1,1）
        g = self.gate(fea)

        # 残差主干
        for blk in self.blocks:
            if self.checkpoint and self.training:
                fea = torch.utils.checkpoint.checkpoint(blk, fea)
            else:
                fea = blk(fea)

        # 输出残差
        delta = self.out_conv(fea) * self.scale          # -> (B,1,H,W)
        out = torch.clamp(identity + delta, 0.0, 1.0)

        # 可学习混合：enhanced = g * network + (1-g) * identity
        enhanced = g * out + (1 - g) * identity
        return enhanced

    # ----- 部署友好接口 -----
    @torch.no_grad()
    def enhance(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: uint8 [0,255] or float32 [0,1] (B,C,H,W)
        return: uint8 [0,255]
        """
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = img.to(next(self.parameters()).device)
        self.eval()
        out = self.forward(img)
        out = (out * 255.0).round().clamp(0, 255).to(torch.uint8)
        return out


# 修改点 1: 增加 gva_enhancer 参数
def test_single_volume_full_enhance_BSUI(image, label, model, gva_enhancer, classes, patch_size=[256, 256]):
    # image shape: [B, C, H, W] -> 通常是 [1, 1, H, W]
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    prediction = np.zeros_like(label)

    # [逻辑修复]: 不要在这里初始化 GVA2，必须使用外面传进来的模型！
    # gva_enhancer = GVA2(...).cuda() <-- 删除这行

    for ind in range(image.shape[0]):
        # 修改点 2: 明确取出第 0 个通道，确保 slice 是 2D (H, W)
        # image[ind] 是 (C, H, W)，也就是 (1, H, W)
        # image[ind, 0, :, :] 才是 (H, W)
        slice = image[ind, 0, :, :]

        x, y = slice.shape[0], slice.shape[1]

        # 现在 slice 是 2D，zoom 就不会报错了
        slice_zoomed = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

        # 转回 Tensor
        # 注意: 如果你在 Dataset 里已经归一化过了，这里就不要再 /255 了
        # 为了稳妥，这里假设输入已经是 [0,1] 或者根据需要调整
        # 如果 dataset 出来就是 [0,1] float，这里不用动。如果 dataset 出来是 [0,255]，需要除以 255
        input_t = torch.from_numpy(slice_zoomed).float()

        # 增加维度 [H, W] -> [1, 1, H, W]
        input_d = input_t.unsqueeze(0).unsqueeze(0).cuda()

        model.eval()
        gva_enhancer.eval()  # 确保增强器也是 eval 模式

        with torch.no_grad():
            # 使用传入的训练好的 gva_enhancer
            img_enhanced = gva_enhancer(input_d)
            output = model(img_enhanced)

            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]

            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)  # [H, W]
            out = out.cpu().detach().numpy()

            # 还原尺寸
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    metric_list = []
    # 二分类 classes=2，这里 range(1, 2) 只计算前景类
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_full_enhance(image, label, model, classes, enhancer, patch_size=[256, 256]):
    """
    Args:
        image: 原始图像 volume [D, H, W]
        label: 原始标签 volume [D, H, W]
        model: 分割网络 (UNet)
        classes: 类别数
        enhancer: 训练好的 GVA2 增强网络 <--- 关键参数
        patch_size: 网络输入的 patch 大小
    """
    # 1. 准备数据，转为 numpy
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()

    # 处理可能的 Batch 维度 [1, D, H, W] -> [D, H, W]
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(label.shape) == 4:
        label = label.squeeze(0)

    prediction = np.zeros_like(label)

    # 2. 切换评估模式
    model.eval()
    if enhancer is not None:
        enhancer.eval()

    # 3. 逐切片处理
    for ind in range(image.shape[0]):
        slice_data = image[ind, :, :]
        x, y = slice_data.shape

        # --- 预处理 ---
        # 改进点：图像缩放使用 order=3 (Bicubic) 效果更好，保留更多细节
        # 注意：如果是比较旧的 scipy 版本，zoom 可能不接受 sequence 作为 zoom factor，视情况调整
        slice_res = zoom(slice_data, (patch_size[0] / x, patch_size[1] / y), order=3)

        # 归一化与转 Tensor
        # 假设原始数据是 [0, 255] 或已归一化，这里为了稳健做个判断
        input_tensor = torch.from_numpy(slice_res).float()
        if input_tensor.max() > 1:
            input_tensor = input_tensor / 255.0

        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, H, W]

        with torch.no_grad():
            # --- 关键步骤：使用训练好的 Enhancer ---
            if enhancer is not None:
                # GVA2 前向
                input_tensor = enhancer(input_tensor)

            # --- 分割预测 ---
            output = model(input_tensor)

            # 处理 Deep Supervision 返回 Tuple 的情况
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]

            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            # --- 后处理 ---
            # 预测结果还原回原始尺寸 (Mask 必须用 order=0 防止产生小数类别)
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    # 4. 计算指标
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list

def test_single_volume (image, label, model, classes, patch_size=[256, 256]):
    """
    Args:
        image: 原始图像 volume [D, H, W]
        label: 原始标签 volume [D, H, W]
        model: 分割网络 (UNet)
        classes: 类别数
        enhancer: 训练好的 GVA2 增强网络 <--- 关键参数
        patch_size: 网络输入的 patch 大小
    """
    # 1. 准备数据，转为 numpy
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()

    # 处理可能的 Batch 维度 [1, D, H, W] -> [D, H, W]
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(label.shape) == 4:
        label = label.squeeze(0)

    prediction = np.zeros_like(label)

    # 2. 切换评估模式
    model.eval()


    # 3. 逐切片处理
    for ind in range(image.shape[0]):
        slice_data = image[ind, :, :]
        x, y = slice_data.shape

        # --- 预处理 ---
        # 改进点：图像缩放使用 order=3 (Bicubic) 效果更好，保留更多细节
        # 注意：如果是比较旧的 scipy 版本，zoom 可能不接受 sequence 作为 zoom factor，视情况调整
        slice_res = zoom(slice_data, (patch_size[0] / x, patch_size[1] / y), order=3)

        # 归一化与转 Tensor
        # 假设原始数据是 [0, 255] 或已归一化，这里为了稳健做个判断
        input_tensor = torch.from_numpy(slice_res).float()
        if input_tensor.max() > 1:
            input_tensor = input_tensor / 255.0

        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, H, W]

        with torch.no_grad():
            # --- 关键步骤：使用训练好的 Enhancer ---


            # --- 分割预测 ---
            output = model(input_tensor)

            # 处理 Deep Supervision 返回 Tuple 的情况
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]

            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            # --- 后处理 ---
            # 预测结果还原回原始尺寸 (Mask 必须用 order=0 防止产生小数类别)
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    # 4. 计算指标
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list

# import torch.nn.functional as F
# def test_single_volume(image, label, model, classes, patch_size=[224, 224], test_save_path=None, case=None,
#                        z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#
#     # 这里的 image 输入通常是 [C, H, W] 或 [H, W]
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         # 如果是 3D volume 切片处理逻辑... (此处省略，假设你是 2D 任务)
#         # 如果你的输入确实是 3D 的，请保持原有的切片循环逻辑，但在送入 model 前 resize
#         pass
#     else:
#         # === 针对 2D 图片的处理逻辑 ===
#         input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()  # [1, 1, H, W]
#
#         # 1. 获取原始尺寸
#         _, _, h, w = input_tensor.shape
#
#         # 2. 如果尺寸不符合 224x224，进行插值缩放
#         if h != patch_size[0] or w != patch_size[1]:
#             input_tensor = F.interpolate(input_tensor, size=patch_size, mode='bilinear', align_corners=False)
#
#         # 3. 模型推理
#         model.eval()
#         with torch.no_grad():
#             output = model(input_tensor)
#
#             # 4. 如果之前缩放过，现在还原回原始尺寸
#             if h != patch_size[0] or w != patch_size[1]:
#                 output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
#
#             out_argmax = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
#
#         prediction = out_argmax.cpu().detach().numpy()
#
#     # 计算指标
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))  # 假设你有名为 calculate_metric_percase 的函数
#
#     return metric_list
def test_single_volume_color(image, label, model, classes):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice = torch.from_numpy(slice) / 255
        slice = slice.permute(2, 0, 1)
        input_d = slice.unsqueeze(0).cuda()
        model.eval()
        with torch.no_grad():
            output = model(input_d)
            if len(output) > 1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
            pred = out.cpu().detach().numpy()
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
