
import torch
import numpy as np
from scipy.stats import pearsonr

import utils.utils as tool
from data.config import get_dataset_path

from nets.dataset import get_dataset
from nets.models import Generator

from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

rcParams['font.family'] = 'Times New Roman'
rcParams["mathtext.fontset"] = 'cm'


def load_dataset(args):
    (path_train, path_test, entire_mean, entire_std) = get_dataset_path(
        args.dataset_name
    )
    args.entire_mean = [float(i) for i in entire_mean.split(',')]
    args.entire_std = [float(i) for i in entire_std.split(',')]
    dataset_test = get_dataset(
        path_test,
        args.data_size,
        args.entire_mean,
        args.entire_std,
        args.dataset_name.split('_')[0]
    )
    return dataset_test, args


def load_gen(args, curr_scale, device=torch.device("cuda")):
    g_ema = Generator(
        args.channel,
        args.noise_dim,
        kernelSize_toRGB=args.kernelSize,
        residual_blocks=args.residual_blocks,
        residual_scaling=args.residual_scaling,
    ).to(device)
    checkpoint = torch.load(
        args.log_folder + '/checkpoint/LAG_scale_' + str(curr_scale) + '.pt',
        map_location=device
    )
    g_ema.load_state_dict(checkpoint['gen_ema_state_dict'])
    return g_ema


def generate_fake_image(dataset_test, indices, curr_scale, model, device=torch.device("cuda"), ncand=3):
    samples_list = [dataset_test[i] for i in indices]
    hires = torch.stack(samples_list).to(device)
    lores = tool.upscale(hires, args.max_scale)

    n, _, h, w = lores.shape
    eps_size = [n, args.noise_dim, h, w]

    with torch.no_grad():
        fake = torch.cat(
            [
                model(
                    lores,
                    torch.normal(
                        mean=0,
                        std=y / (ncand - 1) if ncand > 1 else 0,
                        size=eps_size,
                        device=device,
                    ),
                    curr_scale,
                    1,
                )
                for y in range(ncand)
            ],
            dim=3,
        )

    # stretch to origianl data range
    hires = tool.normalize(hires, args.entire_mean, args.entire_std, True)
    fake = tool.normalize(fake, args.entire_mean, args.entire_std, True)

    # increase image size for visualization
    lores = tool.downscale(tool.upscale(hires, args.max_scale), args.max_scale)
    fake = tool.downscale(fake, args.max_scale // curr_scale)
    hires = tool.downscale(hires, args.max_scale // curr_scale)

    return hires, lores, fake


def plot_sample_images(
    args, g_ema, dataset_test, current_scale, sample_indices: list = [], ncand=3, add_xlab=False
):
    assert len(sample_indices) < 2, 'one image at a time'
    sample_indices = (
        sample_indices if sample_indices else torch.randint(
            0, len(dataset_test), (1,))
    )
    print("sample_indices: ", sample_indices)
    data_type = args.dataset_name.split('_')[0]
    color_map = 'viridis' if data_type == 'Wind' else 'inferno'
    eval_folder = args.log_folder + '/eval'
    tool.set_random_seed(666)

    # generate fake images
    hires, lores, fake = generate_fake_image(
        dataset_test, sample_indices, curr_scale, g_ema, ncand=ncand
    )
    lores = lores.cpu().squeeze()
    hires = hires.cpu().squeeze()
    fake = fake.cpu().squeeze()

    subplot_size = 6
    images_num = ncand + 2  # ncand + LR + GT
    rows = args.channel
    cols = images_num + 1  # images + 1 colorbar
    image_size = hires.shape[2]
    fontsize = 30
    fontsize_xlab = 30

    fig = plt.figure(figsize=(cols * subplot_size, rows * subplot_size))
    gs = gridspec.GridSpec(
        rows,
        cols,
        width_ratios=[1 for _ in range(images_num)] + [0.05],
        height_ratios=[1 for _ in range(args.channel)],
    )

    if add_xlab:
        xlab_list = ["a", "b", "c", "d", "e", "f",
                     "g", "h", "i", "j", "k", "l", "m", "n"]
        xlab_idx = 0
    for c in range(args.channel):
        vmin0, vmax0 = (hires[c, :].min(), hires[c, :].max())
        # real_lr
        ax_lr = fig.add_subplot(gs[c, 0])
        ax_lr.imshow(
            lores[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map
        )
        ax_lr.set(xticks=[], yticks=[])
        if add_xlab:
            ax_lr.set_xlabel(
                "("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        if data_type == 'Wind':
            ax_lr.set_ylabel('V' if c else 'U', fontsize=fontsize_xlab)
        elif data_type == 'Solar':
            ax_lr.set_ylabel('DHI' if c else 'DNI', fontsize=fontsize_xlab)
        if c == 0:
            ax_lr.set_title('LR', {'fontsize': fontsize})

        for s in range(ncand):
            ax = fig.add_subplot(gs[c, s + 1])
            ax.imshow(
                fake[c, :, image_size * s: image_size * (s + 1)],
                vmin=vmin0,
                vmax=vmax0,
                cmap=color_map
            )
            ax.set(xticks=[], yticks=[])
            if add_xlab:
                ax.set_xlabel(
                    "("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
                xlab_idx += 1
            if c == 0:
                ax.set_title(
                    "$z\sim\mathcal{N}(0," +
                    str(int((s / (ncand - 1))**2)) + ")$",
                    fontsize=fontsize,
                )

        # real_hr
        ax_hr = fig.add_subplot(gs[c, images_num - 1])
        ax_hr.set(xticks=[], yticks=[])
        if add_xlab:
            ax_hr.set_xlabel(
                "("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        if c == 0:
            ax_hr.set_title('Ground Truth', {'fontsize': fontsize})
        im = ax_hr.imshow(
            hires[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map
        )

        # colorbar
        if data_type == 'Wind':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(label='$m/s$', size=fontsize_xlab)
        elif data_type == 'Solar':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(label='$W/m^2$', size=fontsize_xlab)
        else:
            raise ValueError('Currently only supports Wind and Solar')

    save_path = eval_folder + '/test_samples_' + \
        str(sample_indices) + '_scale' + str(current_scale) + '.png'
    # plt.show()
    plt.subplots_adjust(wspace=0, hspace=0.2, right=0.8)
    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    plt.close()
    print("Saved:", save_path)


def plot_sample_images_different_scale(
    args, dataset_test, sample_indices: list = []
):
    assert len(sample_indices) < 2, 'one image at a time'
    sample_indices = (
        sample_indices if sample_indices else torch.randint(
            0, len(dataset_test), (1,))
    )
    data_type = args.dataset_name.split('_')[0]
    color_map = 'viridis' if data_type == 'Wind' else 'inferno'
    eval_folder = args.log_folder + '/eval'

    _, _, fake_4 = generate_fake_image(
        dataset_test, sample_indices, 4, load_gen(args, 4), ncand=1
    )
    fake_4 = fake_4.cpu().squeeze()

    _, _, fake_8 = generate_fake_image(
        dataset_test, sample_indices, 8, load_gen(args, 8), ncand=1
    )
    fake_8 = fake_8.cpu().squeeze()

    _, _, fake_16 = generate_fake_image(
        dataset_test, sample_indices, 16, load_gen(args, 16), ncand=1
    )
    fake_16 = fake_16.cpu().squeeze()

    _, _, fake_32 = generate_fake_image(
        dataset_test, sample_indices, 32, load_gen(args, 32), ncand=1
    )
    fake_32 = fake_32.cpu().squeeze()

    hires, lores, fake_64 = generate_fake_image(
        dataset_test, sample_indices, 64, load_gen(args, 64), ncand=1
    )
    fake_64 = fake_64.cpu().squeeze()

    fake_list = [fake_4, fake_8, fake_16, fake_32, fake_64]

    lores = lores.cpu().squeeze()
    hires = hires.cpu().squeeze()

    subplot_size = 6
    images_num = 7  # LR + 4x + 8x + 16x + 32x + 64x + GT
    rows = args.channel
    cols = images_num + 1  # images + 1 colorbar
    fontsize = 30
    fontsize_xlab = 30

    fig = plt.figure(figsize=(cols * subplot_size, rows * subplot_size))
    gs = gridspec.GridSpec(
        rows,
        cols,
        width_ratios=[1 for _ in range(images_num)] + [0.05],
        height_ratios=[1 for _ in range(args.channel)],
    )

    xlab_list = ["a", "b", "c", "d", "e", "f",
                 "g", "h", "i", "j", "k", "l", "m", "n"]
    xlab_idx = 0
    for c in range(args.channel):
        vmin0, vmax0 = (hires[c, :].min(), hires[c, :].max())
        # real_lr
        ax_lr = fig.add_subplot(gs[c, 0])
        ax_lr.imshow(
            lores[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map
        )
        ax_lr.set(xticks=[], yticks=[])
        ax_lr.set_xlabel("("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
        xlab_idx += 1
        if data_type == 'Wind':
            ax_lr.set_ylabel('V' if c else 'U', fontsize=fontsize_xlab)
        elif data_type == 'Solar':
            ax_lr.set_ylabel('DHI' if c else 'DNI', fontsize=fontsize_xlab)
        if c == 0:
            ax_lr.set_title('LR', {'fontsize': fontsize})

        for s, scale_title in enumerate([r"$4\times$ SR", r"$8\times$ SR", r"$16\times$ SR", r"$32\times$ SR", r"$64\times$ SR"]):
            ax = fig.add_subplot(gs[c, s + 1])
            ax.imshow(
                fake_list[s][c, :, :],
                vmin=vmin0,
                vmax=vmax0,
                cmap=color_map
            )
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel("("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
            if c == 0:
                ax.set_title(
                    scale_title,
                    fontsize=fontsize
                )

        # real_hr
        ax_hr = fig.add_subplot(gs[c, images_num - 1])
        ax_hr.set(xticks=[], yticks=[])
        ax_hr.set_xlabel("("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
        xlab_idx += 1
        if c == 0:
            ax_hr.set_title('Ground Truth', {'fontsize': fontsize})
        im = ax_hr.imshow(
            hires[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map
        )

        # colorbar
        if data_type == 'Wind':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(label='$m/s$', size=fontsize_xlab)
        elif data_type == 'Solar':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(label='$W/m^2$', size=fontsize_xlab)
        else:
            raise ValueError('Currently only supports Wind and Solar')

    save_path = eval_folder + '/test_samples_' + \
        str(sample_indices) + '_different_scales.png'
    # plt.show()
    plt.subplots_adjust(wspace=0, hspace=0.2, right=0.8)
    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    plt.close()
    print("Saved:", save_path)


def plot_repeated_mean_std(
    args,
    g_ema,
    dataset_test,
    curr_scale,
    sample_indices: list = [],
    device=torch.device("cuda"),
    repeat_num=50,
    add_xlab=False
):
    assert len(sample_indices) < 2, 'one image at a time'
    sample_indices = (
        sample_indices if sample_indices else torch.randint(
            0, len(dataset_test), (1,))
    )
    print("sample_indices: ", sample_indices)
    data_type = args.dataset_name.split('_')[0]
    color_map = 'viridis' if data_type == 'Wind' else 'inferno'
    eval_folder = args.log_folder + '/eval'
    tool.set_random_seed(666)

    # generate fake images
    samples_list = [dataset_test[i] for i in sample_indices]
    x = torch.stack(samples_list).to(device)
    lores = tool.upscale(x, args.max_scale)
    hires = tool.upscale(x, args.max_scale // curr_scale)
    hires = tool.normalize(hires, args.entire_mean,
                           args.entire_std, True).cpu().squeeze()

    n, _, h, w = lores.shape
    eps_size = [n, args.noise_dim, h, w]
    fake = torch.empty(
        (repeat_num, args.channel, h * curr_scale, w * curr_scale)
    )
    with torch.no_grad():
        for i in range(repeat_num):
            fake[i] = g_ema(
                lores,
                torch.normal(
                    mean=0,
                    std=1,
                    size=eps_size,
                    device=device,
                ),
                curr_scale,
                1,
            )
            # stretch to origianl data range
            fake[i] = tool.normalize(
                fake[i], args.entire_mean, args.entire_std, True)
    lores = tool.upscale(hires, curr_scale)
    lores = tool.downscale(lores, curr_scale).cpu().squeeze()

    subplot_size = 6
    images_num = 4  # LR + mean + std + GT
    rows = args.channel
    cols = images_num + 1  # images + 1 colorbar
    fontsize = 30
    fontsize_xlab = 30

    fig = plt.figure(figsize=(cols * subplot_size, rows * subplot_size))
    gs = gridspec.GridSpec(
        rows,
        cols,
        width_ratios=[1 for _ in range(images_num)] + [0.05],
        height_ratios=[1 for _ in range(args.channel)],
    )
    if add_xlab:
        xlab_list = ["a", "b", "c", "d", "e", "f",
                     "g", "h", "i", "j", "k", "l", "m", "n"]
        xlab_idx = 0

    for c in range(args.channel):
        vmin0, vmax0 = (hires[c, :].min(), hires[c, :].max())
        # real_lr
        ax_lr = fig.add_subplot(gs[c, 0])
        ax_lr.imshow(
            lores[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map
        )
        ax_lr.set(xticks=[], yticks=[])
        if add_xlab:
            ax_lr.set_xlabel(
                "("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        if data_type == 'Wind':
            ax_lr.set_ylabel('V' if c else 'U', fontsize=fontsize_xlab)
        elif data_type == 'Solar':
            ax_lr.set_ylabel('DHI' if c else 'DNI', fontsize=fontsize_xlab)
        if c == 0:
            ax_lr.set_title('LR', {'fontsize': fontsize})

        ax_mean = fig.add_subplot(gs[c, 1])
        ax_mean.imshow(
            fake[:, c, :, :].mean(0),
            vmin=vmin0,
            vmax=vmax0,
            cmap=color_map
        )
        ax_mean.set(xticks=[], yticks=[])
        if add_xlab:
            ax_mean.set_xlabel(
                "("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        if c == 0:
            ax_mean.set_title(
                r"Mean",
                fontsize=fontsize
            )

        ax_std = fig.add_subplot(gs[c, 2])
        ax_std.imshow(
            fake[:, c, :, :].std(0),
            vmin=vmin0,
            vmax=vmax0,
            cmap=color_map
        )
        ax_std.set(xticks=[], yticks=[])
        if add_xlab:
            ax_std.set_xlabel(
                "("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        if c == 0:
            ax_std.set_title(
                r"Standard deviation",
                fontsize=fontsize,
            )

        # real_hr
        ax_hr = fig.add_subplot(gs[c, images_num - 1])
        ax_hr.set(xticks=[], yticks=[])
        if add_xlab:
            ax_hr.set_xlabel(
                "("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        if c == 0:
            ax_hr.set_title('Ground Truth', {'fontsize': fontsize})
        im = ax_hr.imshow(
            hires[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map
        )

        # colorbar
        if data_type == 'Wind':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(label='$m/s$', size=fontsize_xlab)
        elif data_type == 'Solar':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(label='$W/m^2$', size=fontsize_xlab)
        else:
            raise ValueError('Currently only supports Wind and Solar')

    # plt.show()
    # plt.title(image_type)
    plt.subplots_adjust(wspace=0, hspace=0.2, right=0.8)
    save_path = eval_folder + '/test_samples_' + \
        str(sample_indices) + '_scale' + str(curr_scale) + \
        '_repeat' + str(repeat_num) + '_mean_std.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    plt.close()
    print("Saved:", save_path)


def plot_mass_consistency(
    args, g_ema, dataset_test, current_scale, sample_indices: list = [], ncand=2, repeat=1, device=torch.device("cuda"), add_xlab=False
):
    assert len(sample_indices) < 2, 'one image at a time'
    sample_indices = (
        sample_indices if sample_indices else torch.randint(
            0, len(dataset_test), (1,))
    )
    print("sample_indices: ", sample_indices)
    data_type = args.dataset_name.split('_')[0]
    color_map = 'viridis' if data_type == 'Wind' else 'inferno'
    eval_folder = args.log_folder + '/eval'
    tool.set_random_seed(666)

    # generate fake images
    fake_repeat = torch.empty(
        (repeat, args.channel, 8*curr_scale, 8*curr_scale*ncand), dtype=torch.float64)
    for i in range(repeat):
        hires, _, fake = generate_fake_image(
            dataset_test, sample_indices, curr_scale, g_ema, device, ncand=ncand
        )
        if i == repeat-1:
            hires = hires.cpu().squeeze()
        fake_repeat[i] = fake.cpu().squeeze()
    fake = fake_repeat.mean(0)
    lores = tool.upscale(hires, curr_scale)

    subplot_size = 6
    images_num = ncand + 2  # ncand + LR + GT
    rows = args.channel
    cols = images_num + 1  # images + 1 colorbar
    image_size = hires.shape[2]
    fontsize = 32
    fontsize_xlab = 30

    fig = plt.figure(figsize=(cols * subplot_size, rows * subplot_size))
    gs = gridspec.GridSpec(
        rows,
        cols,
        wspace=0.2,
        width_ratios=[1 for _ in range(images_num)] + [0.05],
        height_ratios=[1 for _ in range(args.channel)],
    )

    if add_xlab:
        xlab_list = ["a", "b", "c", "d", "e", "f",
                     "g", "h", "i", "j", "k", "l", "m", "n"]
        xlab_idx = 0
    for c in range(args.channel):
        vmin0, vmax0 = (hires[c, :].min(), hires[c, :].max())
        # real_lr
        ax_lr = fig.add_subplot(gs[c, 0])
        ax_lr.imshow(lores[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map)
        ax_lr.set(xticks=[], yticks=[])
        ax_lr.set(xticks=[], yticks=[])
        if add_xlab:
            ax_lr.set_xlabel("Correlation coefficient: \n(" +
                             xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        else:
            ax_lr.set_xlabel("Correlation coefficient:",
                             fontsize=fontsize_xlab)

        if data_type == 'Wind':
            ax_lr.set_ylabel('V' if c else 'U', fontsize=fontsize_xlab)
        elif data_type == 'Solar':
            ax_lr.set_ylabel('DHI' if c else 'DNI', fontsize=fontsize_xlab)
        if c == 0:
            ax_lr.set_title('LR', {'fontsize': fontsize})

        data_x = lores[c, :].flatten()
        # add 45 degree line
        x = np.linspace(data_x.min(), data_x.max())

        for s in range(ncand):
            ax = fig.add_subplot(gs[c, s + 1])
            data_y = tool.upscale(fake[c, :, image_size * s: image_size * (s + 1)]
                                  [None, None, :], curr_scale).squeeze().flatten()

            # calculate Pearson's correlation
            corr, _ = pearsonr(data_x, data_y)

            ax.scatter(data_x, data_y, marker=".",
                       edgecolors='none', zorder=2, s=80)
            ax.plot(x, x, color='red', ls='-', lw=0.5, zorder=1)

            ax.set(xticks=[], yticks=[])
            if add_xlab:
                ax.set_xlabel(str(round(corr, 5))+"\n(" +
                              xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
                xlab_idx += 1
            else:
                ax.set_xlabel(str(round(corr, 5)), fontsize=fontsize_xlab)
            if c == 0:
                ax.set_title(
                    "$z\sim\mathcal{N}(0," + str(int((s / (ncand - 1))**2)
                                                 ) + ")$", {'fontsize': fontsize}
                )

        # real_hr
        ax_hr = fig.add_subplot(gs[c, images_num - 1])
        ax_hr.set(xticks=[], yticks=[])
        if add_xlab:
            ax_hr.set_xlabel(
                "\n("+xlab_list[xlab_idx]+")", fontsize=fontsize_xlab)
            xlab_idx += 1
        if c == 0:
            ax_hr.set_title('Ground Truth', {'fontsize': fontsize})
        im = ax_hr.imshow(hires[c, :], vmin=vmin0, vmax=vmax0, cmap=color_map)

        # colorbar
        if data_type == 'Wind':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(
                label='$m/s$', size=fontsize_xlab
            )
        elif data_type == 'Solar':
            cbar = fig.colorbar(im, cax=fig.add_subplot(gs[c, images_num]))
            cbar.ax.tick_params(labelsize=fontsize_xlab, length=10)
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(4)
            cbar.set_label(
                label='$W/m^2$', size=fontsize_xlab
            )
        else:
            raise ValueError('Currently only supports Wind and Solar')

    # plt.show()
    save_path = eval_folder + '/test_samples_' + str(sample_indices) + '_scale' + str(
        current_scale) + '_mass_consistency_repeat'+str(repeat)+'_ncand'+str(ncand)+'.png'
    plt.subplots_adjust(wspace=0, hspace=0.21, right=0.8)
    plt.savefig(save_path, bbox_inches='tight', dpi=650)
    plt.close()
    print("Saved:", save_path)


if __name__ == '__main__':
    args_paths = [
        'results/Solar/Solar_07-14_bs1_epoch15_lr4e-3_64X/args/args.json',
        'results/Wind/Wind_07-14_bs16_epoch30_lr2_64X/args/args.json'
    ]

    test_scales = [64]

    whether_plot_sample_images = True
    whether_plot_mass_consistency = True
    whether_plot_sample_images_different_scale = True

    repeat_num = 1000
    whether_plot_repeated_mean_std = True

    sample_indices = [0]
    for args_path in args_paths:
        args, _ = tool.get_config_from_json(args_path)
        args.log_folder = '/'.join(args_path.split('/')[:-2])

        dataset_test, args = load_dataset(args)

        if whether_plot_sample_images_different_scale:
            plot_sample_images_different_scale(
                args,
                dataset_test,
                sample_indices=sample_indices
            )

        for curr_scale in test_scales:
            g_ema = load_gen(args, curr_scale)

            if whether_plot_sample_images:
                plot_sample_images(
                    args,
                    g_ema,
                    dataset_test,
                    curr_scale,
                    sample_indices=sample_indices,
                    ncand=2
                )

            if whether_plot_repeated_mean_std:
                plot_repeated_mean_std(
                    args,
                    g_ema,
                    dataset_test,
                    curr_scale,
                    sample_indices=sample_indices,
                    repeat_num=repeat_num
                )

            if whether_plot_mass_consistency:
                plot_mass_consistency(
                    args,
                    g_ema,
                    dataset_test,
                    curr_scale,
                    sample_indices
                )
