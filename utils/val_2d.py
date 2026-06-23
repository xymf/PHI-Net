import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pdb



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    # image: [B=1, C=1, H, W] -> squeeze to [H, W]
    # label: [B=1, H, W] -> squeeze to [H, W]
    image = image.squeeze().cpu().detach().numpy()  # [H, W]
    label = label.squeeze().cpu().detach().numpy()  # [H, W]

    x, y = image.shape
    # Resize to patch_size
    if x != patch_size[0] or y != patch_size[1]:
        image_zoom = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    else:
        image_zoom = image

    input_tensor = torch.from_numpy(image_zoom).unsqueeze(0).unsqueeze(0).float().cuda()  # [1,1,H,W]

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)  # [H, W]
        out = out.cpu().detach().numpy()

    # Zoom back to original size
    if x != patch_size[0] or y != patch_size[1]:
        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    else:
        prediction = out

    # Compute metrics for each class
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
# def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
#     image, label = image.squeeze(0).cpu().detach(
#     ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#         model.eval()
#         with torch.no_grad():
#             output = model(input)
#             if len(output)>1:
#                 output = output[0]
#             out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#             prediction[ind] = pred
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))
#     return metric_list

def test_single_volume_cross(image, label, model_l, model_r, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model_r.eval()
        model_l.eval()
        with torch.no_grad():
            output_l = model_l(input)
            output_r = model_r(input)
            output = (output_l + output_r) / 2
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
