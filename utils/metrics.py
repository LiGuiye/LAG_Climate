import torch
import numpy as np


def calc_metrics(
    img_real,
    img_fake,
    metrics: list = ['l2', 'mse', 'rmse', 'swd'],
    forceCPU: bool = True
):
    """
    Calculate different metrics to compare the distance between interpolated images and real images.

    Args:
        image1: a noise-free m x n monochrome image
        image2: its noisy approximation
        metrics: ['l2', 'mse', 'rmse', 'swd']

    Return:
        results: dict
    """
    img_real = img_real.squeeze()
    img_fake = img_fake.squeeze()
    results = {}
    for metric_type in metrics:
        if metric_type == 'l2':
            results['l2'] = ((img_real - img_fake) ** 2).sum().sqrt()
        if metric_type == 'mse':
            results['mse'] = ((img_real - img_fake)**2).mean()
        if metric_type == 'rmse':
            results['rmse'] = ((img_real - img_fake)**2).mean().sqrt()
        if metric_type == 'swd':
            results['swd'] = sliced_wasserstein_cuda(img_real, img_fake).cpu()

    if forceCPU:
        for metric_type in metrics:
            if isinstance(results[metric_type], torch.Tensor):
                results[metric_type] = results[metric_type].cpu()
    return results


# ================================ SWD ================================
def sliced_wasserstein_cuda(A, B, dir_repeats=4, dirs_per_repeat=128, device=torch.device("cuda")):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    results = torch.empty(dir_repeats, device=device)
    A = torch.from_numpy(A).to(device) if not isinstance(A, torch.Tensor) else A.to(device)
    B = torch.from_numpy(B).to(device) if not isinstance(B, torch.Tensor) else B.to(device)
    for repeat in range(dir_repeats):
        dirs = torch.randn(A.shape[-1], dirs_per_repeat, device=device)          # (descriptor_component, direction)
        dirs = torch.divide(dirs, torch.sqrt(torch.sum(torch.square(dirs), dim=0, keepdim=True)))  # normalize descriptor components for each direction
        projA = torch.matmul(A, dirs)                                           # (neighborhood, direction)
        projB = torch.matmul(B, dirs)
        projA = torch.sort(projA, dim=0)[0]                                     # sort neighborhood projections for each direction
        projB = torch.sort(projB, dim=0)[0]
        dists = torch.abs(projA - projB)                                        # pointwise wasserstein distances
        results[repeat] = torch.mean(dists)                                     # average over neighborhoods and directions
    return torch.mean(results)                                                  # average over repeats


def variogram(img1, img2, device=torch.device("cuda"), max_bias='half'):
    """
    Calculate variogram for squared images.
    """
    batch, channel, h, w = img1.shape
    if max_bias == 'half':
        max_bias = round(h / 2)
    else:
        assert max_bias > 0
    vario1 = torch.zeros((max_bias, batch, channel), device=device)
    vario2 = torch.zeros((max_bias, batch, channel), device=device)

    for lag in range(1, max_bias+1):
        valid_num_pixels = w*(h-lag) + h*(w-lag)

        # squared row-wise difference
        r1 = (img1[..., lag:, :]-img1[..., :-lag, :])**2
        r2 = (img2[..., lag:, :]-img2[..., :-lag, :])**2

        # squared column-wise difference
        r3 = (img1[..., :, lag:]-img1[..., :, :-lag])**2
        r4 = (img2[..., :, lag:]-img2[..., :, :-lag])**2

        # Sum along height and width
        vario1[lag-1] = (r1.sum((2, 3)) + r3.sum((2, 3))) / valid_num_pixels
        vario2[lag-1] = (r2.sum((2, 3)) + r4.sum((2, 3))) / valid_num_pixels
    return ((vario1-vario2)**2).mean()