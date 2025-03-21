import os
import time
import json
import copy
import random
import shutil
import logging
import numpy as np
from bunch import Bunch
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F


def get_time():
    return time.time()


def rd(loss, unit=3):
    """Round to given unit"""
    return round(loss if isinstance(loss, (float,int)) else loss.item(), unit)


def get_logger(filename, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger


def ilog2(x):
    """return Integer log2."""
    return int(np.ceil(np.log2(x)))


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def set_random_seed(seed=None, verbose=False):
    '''
    leave seed empty to random set random seed
    '''
    seed = 0 if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if verbose:
        print("Random Seed: ", seed)
    return seed


def print_args(args):
    print('-' * 80)
    for k, v in args.__dict__.items():
        print('%-32s %s' % (k, v))
    print('-' * 80)


def save_args(args, path=None):
    with open(
        os.path.join(args.arg_dir if path is None else path, 'args.json'), 'w'
    ) as f:
        json.dump(vars(args), f, sort_keys=False, indent=4)


def get_config_from_json(json_file):
    """
    Get the config from a json file

    Args:
        json_file
    Return:
        config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def remove_all(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)


def mkdir(dirList):
    """
    make dir for all the path in the dirList
    """
    if type(dirList) == list:
        for dir in dirList:
            os.makedirs(dir, exist_ok=True)
    else:
        os.makedirs(dirList, exist_ok=True)


def get_model(model):
    """
    Retrieve the original network. Use this function
    when you want to modify your network after the initialization
    """
    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        return model.module
    return model


def move_optim(optimizer, device):
    """Move Optimizer to target device"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def scale(data, original_min, original_max, target_min, target_max):
    """
    Min-max stretch
    """
    pixel_min = data.min() if original_min is None else original_min
    pixel_max = data.max() if original_max is None else original_max

    return (data-pixel_min) / max(pixel_max-pixel_min, 1e-5) * (target_max-target_min) + target_min


def normalize_standard(image, mean=None, std=None):
    mean = image.mean() if mean is None else mean
    std = image.std() if std is None else std
    return (image - mean) / max(std, 1e-5)


def inverse_normalize_standard(image, mean=None, std=None):
    return image * max(std, 1e-5) + mean


def normalize(img, entire_mean: list, entire_std: list, inverse: bool = False):
    """Using given mean and std to normalize images.
    If inverse is True, do the inverse process.

    Args:
        images: NCHW or CHW
    Return:
        images
    """
    images = img.clone()

    if images.dim() == 3:
        c, _, _ = images.shape
        for j in range(c):
            if inverse:
                images[j] = inverse_normalize_standard(
                    images[j], entire_mean[j], entire_std[j]
                )
            else:
                images[j] = normalize_standard(images[j], entire_mean[j], entire_std[j])
    elif images.dim() == 4:
        n, c, _, _ = images.shape
        for y in range(n):
            for j in range(c):
                if inverse:
                    images[y][j] = inverse_normalize_standard(
                        images[y][j], entire_mean[j], entire_std[j]
                    )
                else:
                    images[y][j] = normalize_standard(
                        images[y][j], entire_mean[j], entire_std[j]
                    )
    return images


def build_ema(model, device=torch.device("cuda")):
    r"""
    Create and upload a moving average generator.
    """
    model_ema = copy.deepcopy(get_model(model))
    for param in model_ema.parameters():
        param.requires_grad = False
    return model_ema.to(device)


def update_ema(model_ema, model, ema_decay=0.999, device=torch.device("cuda")):
    named_param = dict(get_model(model).named_parameters())
    for k, v in model_ema.named_parameters():
        v.copy_(ema_decay * v + (1 - ema_decay) * named_param[k].to(device))


def check_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_num = torch.cuda.device_count()
        num_workers = gpu_num * 4
        print("multiGPUs" if gpu_num > 1 else "Single GPU")
    else:
        device = torch.device("cpu")
        gpu_num = 0
        print("CPU")
        num_workers = 2
    return device, gpu_num, num_workers


def upscale(feat, scale_factor: int = 2):
    """resolution decrease"""
    if scale_factor == 1:
        return feat
    else:
        if feat.dim() != 4:
            feat = feat.view((*([1]*(4-feat.dim())),*feat.shape))
        return F.avg_pool2d(feat, scale_factor)


def downscale(feat, scale_factor: int = 2, mode: str = 'nearest'):
    """resolution increase"""
    if scale_factor == 1:
        return feat
    else:
        if feat.dim() != 4:
            feat = feat.view((*([1]*(4-feat.dim())),*feat.shape))
        return F.interpolate(
            feat,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=True if mode == 'bicubic' else None,
        )


def save_plot(hires, lores, fake, savePath, data_name="Solar"):
    n, channel, h, w = hires.shape
    w_fake = fake.shape[3] if fake.dim() == 4 else fake.shape[2]
    fake_num = w_fake // w

    subplot_size = 6
    rows = n
    cols = 2 + fake_num  # real_hr, real_lr, fake_hr (e.g., eps0-1)
    color_map = 'viridis' if data_name == 'Wind' else 'inferno'

    for c in range(channel):
        fig, axs = plt.subplots(
            ncols=cols,
            nrows=rows,
            figsize=(cols * subplot_size, rows * subplot_size),
        )
        if rows == 1:
            axs = axs.reshape(1, cols)
        axs[0, 0].set_title('Ground Truth', {'fontsize': 9})
        axs[0, 1].set_title('Input LR', {'fontsize': 9})

        for i in range(n):
            vmin0, vmax0 = (
                hires[i, c, :, :].min(),
                hires[i, c, :, :].max(),
            )
            # real_hr
            im = axs[i, 0].imshow(
                hires[i, c, :, :],
                vmin=vmin0,
                vmax=vmax0,
                cmap=color_map
            )
            fig.colorbar(im, ax=[axs[i, 0]], location='left', fraction=0.046, pad=0.04)
            axs[i, 0].set(xticks=[], yticks=[])

            # real_lr
            im = axs[i, 1].imshow(
                lores[i, c, :, :],
                vmin=vmin0,
                vmax=vmax0,
                cmap=color_map
            )
            axs[i, 1].set(xticks=[], yticks=[])

            # fake_hr
            for idx in range(fake_num):
                im = axs[i, 2 + idx].imshow(
                    fake[i, c, :, (w * idx): (w * (idx + 1))],
                    vmin=vmin0,
                    vmax=vmax0,
                    cmap=color_map
                )
                axs[i, 2 + idx].set(xticks=[], yticks=[])
                if i == 0:
                    axs[0, 2 + idx].set_title(
                        f"Output HR Std{round(idx / (fake_num - 1), 2) if fake_num > 1 else 0}",
                        {'fontsize': 9}
                    )

        plt.savefig(f"{savePath[:-4]}_channel{c}.png", bbox_inches='tight')
        plt.close()
