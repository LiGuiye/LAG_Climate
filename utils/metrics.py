import torch
import numpy as np


def calc_metrics(
    img_real,
    img_fake,
    metrics: list = ['l2', 'mse', 'rmse', 'swd'],
    cuda: bool = True
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
            # L2 Norm
            if cuda:
                l2 = torch.sqrt(torch.sum(torch.square((img_real - img_fake))))
            else:
                l2 = np.sqrt(np.sum((img_real - img_fake) ** 2))
            results['l2'] = l2
        if metric_type == 'mse':
            if cuda:
                mse = torch.mean(torch.square((img_real - img_fake)))
            else:
                mse = np.mean(np.square(img_real - img_fake))
            results['mse'] = mse
        if metric_type == 'rmse':
            # rmse
            if cuda:
                mse = torch.mean(torch.square((img_real - img_fake)))
                rmse = torch.sqrt(mse)
            else:
                mse = np.mean(np.square(img_real - img_fake))
                rmse = np.sqrt(mse)
            results['rmse'] = rmse
        if metric_type == 'swd':
            # swd
            swd = sliced_wasserstein_cuda(img_real, img_fake)
            results['swd'] = swd
    return results


# ================================ SWD ================================
def sliced_wasserstein_cuda(A, B, dir_repeats=4, dirs_per_repeat=128):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    assert A.ndim == 2 and A.shape == B.shape                                   # (neighborhood, descriptor_component)
    device = torch.device("cuda")
    results = torch.empty(dir_repeats, device=torch.device("cpu"))
    A = torch.from_numpy(A).to(device) if not isinstance(A, torch.Tensor) else A.to(device)
    B = torch.from_numpy(B).to(device) if not isinstance(B, torch.Tensor) else B.to(device)
    for repeat in range(dir_repeats):
        dirs = torch.randn(A.shape[1], dirs_per_repeat, device=device)          # (descriptor_component, direction)
        dirs = torch.divide(dirs, torch.sqrt(torch.sum(torch.square(dirs), dim=0, keepdim=True)))  # normalize descriptor components for each direction
        projA = torch.matmul(A, dirs)                                           # (neighborhood, direction)
        projB = torch.matmul(B, dirs)
        projA = torch.sort(projA, dim=0)[0]                                     # sort neighborhood projections for each direction
        projB = torch.sort(projB, dim=0)[0]
        dists = torch.abs(projA - projB)                                        # pointwise wasserstein distances
        results[repeat] = torch.mean(dists)                                     # average over neighborhoods and directions
    return torch.mean(results)                                                  # average over repeats