import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import utils.utils as tool
from utils.metrics import sliced_wasserstein_cuda
from nets.dataset import get_dataset
from nets.models import Discriminator, Generator
from torch import optim
from torch.utils.data import DataLoader
from data.config import get_dataset_path
import os


class LAGTrainer:
    def __init__(self, args):
        super(LAGTrainer, self).__init__()
        self.report_step = args.report_step
        self.log_folder = 'results/' + args.trial_name
        self.device, gpu_num, self.num_workers = tool.check_cuda_availability()
        # transition epochs:  args.epoch[0]
        # stabilization epochs:  args.epoch[1]
        epoch = [int(i) for i in args.epoch.split(',')]
        self.transition_epoch, self.stablization_epoch = epoch
        self.epochs_per_stage = sum(epoch)
        self.load_data(args)
        self.load_model(args)

    def load_data(self, args):
        self.batch_size = args.batch_size
        self.channel = args.channel
        self.sample_indices = args.sample_indices
        self.data_type = args.dataset_name.split('_')[0]
        (path_train, path_test, entire_mean, entire_std) = get_dataset_path(args.dataset_name)
        self.entire_mean = [float(i) for i in entire_mean.split(',')]
        self.entire_std = [float(i) for i in entire_std.split(',')]
        self.dataset_train = get_dataset(
            path_train,
            args.data_size,
            self.entire_mean,
            self.entire_std,
            self.data_type
        )
        self.dataset_length = len(self.dataset_train)

        self.loader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        self.dataset_test = get_dataset(
            path_test,
            args.data_size,
            self.entire_mean,
            self.entire_std,
            self.data_type
        )

    def load_model(self, args):
        self.max_scale = args.max_scale
        self.noise_dim = args.noise_dim
        self.n_critic = args.n_critic
        self.mse_weight = args.mse_weight
        self.curr_scale = self.max_scale if args.memtest else 2
        self.gen = Generator(
            args.channel,
            args.noise_dim,
            kernelSize_toRGB=args.kernelSize,
            residual_blocks=args.residual_blocks,
            residual_scaling=args.residual_scaling,
        )
        self.gen_ema = tool.build_ema(self.gen)
        self.dis = Discriminator(
            args.channel,
            kernelSize_fromRGB=args.kernelSize,
            residual_blocks=args.residual_blocks,
        )

        # optimizer
        self.g_optim = optim.Adam(
            tool.get_original_model(self.gen).parameters(),
            lr=args.lr,
            betas=(0.0, 0.99),
        )
        self.d_optim = optim.Adam(
            tool.get_original_model(self.dis).parameters(),
            lr=args.lr,
            betas=(0.0, 0.99),
        )

        if args.reset:
            print("Training from scratch!")
            tool.remove_everything(self.log_folder)
            tool.mkdir(
                [
                    self.log_folder,
                    self.log_folder + '/checkpoint',
                    self.log_folder + '/eval',
                    self.log_folder + '/args'
                ]
            )
            tool.save_args(args, self.log_folder + '/args')
            self.curr_epoch = 0
        else:
            # try to load previous weights
            if os.path.exists(self.log_folder + '/checkpoint/tmp'):
                ckpt_path = glob(self.log_folder + '/checkpoint/tmp/*.pt')[0]
                self.curr_epoch = int(ckpt_path.split('.')[0].split('_')[-1])
                self.curr_scale = int(ckpt_path.split('.')[0].split('_')[-3])
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=self.device,
                )
            else:
                ckpt_path = self.log_folder + '/checkpoint'
                scale_max = max(
                    [
                        int(i.split('.')[0].split('_')[-1])
                        for i in glob(ckpt_path + '/*.pt')
                    ]
                )
                checkpoint = torch.load(
                    ckpt_path + '/LAG_scale_' + str(scale_max) + '.pt',
                    map_location=self.device,
                )
                assert self.curr_scale < self.max_scale, 'Model have already been trained'
                self.curr_scale = scale_max * 2
                self.curr_epoch = 0

            print("Training continued!")

            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_ema.load_state_dict(checkpoint['gen_ema_state_dict'])
            self.dis.load_state_dict(checkpoint['dis_state_dict'])
            self.g_optim.load_state_dict(checkpoint['optimizer_gen_state_dict'])
            self.d_optim.load_state_dict(checkpoint['optimizer_dis_state_dict'])
            tool.move_optimizer(self.g_optim, self.device)
            tool.move_optimizer(self.d_optim, self.device)

        # move model to device
        self.gen.to(self.device)
        self.dis.to(self.device)

        tool.print_args(args)

    def calc_gp(self, n, real, fake_eps, lores, alpha):
        """
        calcuate gradient penalty for Dis
        """

        gp_alpha = torch.rand(n, 1, 1, 1).to(self.device)
        mixed = gp_alpha * real.data + (1 - gp_alpha) * fake_eps.detach().data
        mixed.requires_grad = True

        mixed = tool.downscale(
            mixed,
            self.max_scale // self.curr_scale,  # downscale to largest image resolution
        )
        mixed_residual = torch.abs(
            lores
            - tool.upscale(mixed, self.max_scale)  # upscale to lores and calc residual
        )
        mixdown = tool.upscale(
            mixed,
            self.max_scale
            // self.curr_scale,  # upscale to current target image resolution
        )
        mixed_scores = torch.sum(
            torch.mean(self.dis(mixdown, mixed_residual, self.curr_scale, alpha), 1)
        )
        grad_mixed = torch.autograd.grad(
            outputs=mixed_scores, inputs=mixed, create_graph=True
        )[0]
        grad_norm = grad_mixed.view(grad_mixed.size(0), -1).norm(2, dim=1)
        loss_gp = 10 * ((grad_norm - 1) ** 2).mean()

        return loss_gp

    def train_epoch(self):
        start_time = time.time()
        curr_stage_time = None

        total_epochs = tool.ilog2(self.max_scale) * self.epochs_per_stage
        trained_epochs = (
            tool.ilog2(self.curr_scale) - 1
        ) * self.epochs_per_stage + self.curr_epoch
        remain_epochs = total_epochs - trained_epochs

        iters_per_epoch = self.dataset_length // self.batch_size
        iters_per_transition = self.transition_epoch * iters_per_epoch

        if self.curr_epoch == 0:
            alpha = 0
        else:
            iteration = self.curr_epoch * iters_per_epoch + 1
            alpha = min(1, (1 / iters_per_transition) * iteration)

        # for epoch in range(self.epochs_per_stage):
        for epoch in range(remain_epochs):
            if self.curr_epoch == 0 or epoch == 0:
                print(
                    'Scale factor:',
                    self.curr_scale,
                    '/ Total epochs remain: ',
                    total_epochs - trained_epochs,
                )
                print('Transition stage start:')
            if self.curr_epoch == 0:
                alpha = 0
            elif self.curr_epoch == self.transition_epoch:
                print('Stabilization stage start:')

            for i, x in enumerate(self.loader_train):

                x = x.to(self.device)
                lores = tool.upscale(x, self.max_scale)
                real = tool.upscale(x, self.max_scale // self.curr_scale)

                # extra data process during transition stage to stabilize the training
                # 1. `downscale(upscale(real))` to remove detailed information
                # 2. gradually blend them together (alpha gradually increase from 0 to 1)
                real = (1 - alpha) * tool.downscale(tool.upscale(real)) + alpha * real

                ### 1. train Discriminator
                self.dis.zero_grad()
                self.d_optim.zero_grad()

                n, _, h, w = lores.shape
                eps = torch.randn(n, self.noise_dim, h, w, device=self.device)
                fake_eps = self.gen(lores, eps, self.curr_scale, alpha)

                # downscale fake_eps to highest resolution, then upscale to lowest resolution for calc
                lores_fake_eps = tool.upscale(
                    tool.downscale(fake_eps, self.max_scale // self.curr_scale),
                    self.max_scale,
                )
                # move them to latent space
                latent_real = self.dis(
                    real, torch.abs(lores - lores), self.curr_scale, alpha
                )
                latent_fake_eps = self.dis(
                    fake_eps,
                    torch.abs(lores - lores_fake_eps),
                    self.curr_scale,
                    alpha,
                )

                loss_gp = self.calc_gp(n, real, fake_eps, lores, alpha)

                # calc other loss items for Dis
                loss_dreal = -torch.mean(latent_real)
                loss_dfake = torch.mean(latent_fake_eps)

                loss_disc = loss_dfake + loss_dreal + loss_gp

                loss_disc.backward()
                self.d_optim.step()

                ### 2. train Generator
                if (i + 1) % self.n_critic == 0:
                    self.gen.zero_grad()
                    self.g_optim.zero_grad()

                    eps = torch.randn(n, self.noise_dim, h, w, device=self.device)
                    fake_eps = self.gen(lores, eps, self.curr_scale, alpha)
                    fake = self.gen(
                        lores, torch.zeros_like(eps), self.curr_scale, alpha
                    )

                    # upscale fake_eps to lores resolution for calc
                    # -------------
                    lores_fake_eps = tool.upscale(
                        tool.downscale(fake_eps, self.max_scale // self.curr_scale),
                        self.max_scale,
                    )
                    lores_fake = tool.upscale(
                        tool.downscale(fake, self.max_scale // self.curr_scale),
                        self.max_scale,
                    )
                    # -------------

                    # move them to latent space
                    # P(x, y)
                    latent_real = self.dis(
                        real, torch.abs(lores - lores), self.curr_scale, alpha
                    )
                    # P(G(y, z), y)
                    latent_fake_eps = self.dis(
                        fake_eps,
                        torch.abs(lores - lores_fake_eps),
                        self.curr_scale,
                        alpha,
                    )
                    # P(G(y, 0), y)
                    latent_fake = self.dis(
                        fake,
                        torch.abs(lores - lores_fake),
                        self.curr_scale,
                        alpha,
                    )

                    # calc loss items for Gen
                    loss_gfake = -torch.mean(latent_fake_eps)
                    loss_gmse = F.mse_loss(latent_real, latent_fake)
                    loss_gen = loss_gfake + self.mse_weight * loss_gmse

                    loss_gen.backward()
                    self.g_optim.step()
                    tool.update_ema(self.gen_ema, self.gen)

                # log
                trained_step = trained_epochs * iters_per_epoch + i + 1
                if trained_step % self.report_step == 0:
                    print(f"Loss_G: gfake {round(loss_gfake.item(),3)}, gmse {round(loss_gmse.item(),3)}, total_loss {round(loss_gen.item(),3)} | " +
                          f"Loss_D: dreal {round(loss_dreal.item(),3)}, dfake {round(loss_dfake.item(),3)}, gp {round(loss_gp.item(),3)}, total_loss {round(loss_disc.item(),3)}")

                    hires, _, fake = self.gen_test_samples(
                        self.gen_ema,
                        alpha,
                        ncand=1,
                        return_original_range=False,
                    )
                    validation_metrics = self.validation_loss(hires, fake)
                    print(
                        'Validation: MSE',
                        np.round(validation_metrics["mse"], 3),
                        'SWD',
                        np.round(validation_metrics["swd"], 3),
                    )

                # update alpha and iteration count
                iteration = self.curr_epoch * iters_per_epoch + i + 1
                alpha = min(1, (1 / iters_per_transition) * iteration)

            trained_epochs += 1
            self.curr_epoch += 1
            if self.curr_epoch == self.epochs_per_stage:
                self.save_test_samples(alpha)
                self.save_model()

                finish_time = (
                    time.time() - start_time
                    if curr_stage_time is None
                    else time.time() - curr_stage_time
                )
                curr_stage_time = time.time()
                total_time_print = 'Current scale training Finished. Took {:.4f} minutes or {:.4f} hours to complete.'.format(
                    finish_time / 60, finish_time / 3600
                )
                print(total_time_print)

                # new transition and training stages begin
                self.curr_scale *= 2
                self.curr_epoch = 0
            else:
                self.save_model(self.curr_epoch)

        # print and save total time consuming
        finish_time = time.time() - start_time
        total_time_print = 'Training Finished. Took {:.4f} minutes or {:.4f} hours to complete.'.format(
            finish_time / 60, finish_time / 3600
        )
        print(total_time_print)

    def save_model(self, curr_epoch=None):
        if curr_epoch is None:
            save_path = (
                self.log_folder
                + '/checkpoint/LAG_scale_'
                + str(self.curr_scale)
                + '.pt'
            )
            tool.remove_everything(self.log_folder + '/checkpoint/tmp')
        else:
            tool.remove_everything(self.log_folder + '/checkpoint/tmp')
            tool.mkdir([self.log_folder + '/checkpoint/tmp'])
            save_path = (
                self.log_folder
                + '/checkpoint/tmp/LAG_scale_'
                + str(self.curr_scale)
                + '_epoch_'
                + str(curr_epoch)
                + '.pt'
            )
        save_weights = {
            'gen_state_dict': tool.get_original_model(self.gen).state_dict(),
            'gen_ema_state_dict': tool.get_original_model(self.gen_ema).state_dict(),
            'dis_state_dict': tool.get_original_model(self.dis).state_dict(),
            'optimizer_gen_state_dict': self.g_optim.state_dict(),
            'optimizer_dis_state_dict': self.d_optim.state_dict(),
        }
        torch.save(save_weights, save_path)

    def validation_loss(self, hires, fake):
        """
        calculate mse and swd for validation
        """
        validation_metrics = {}
        validation_metrics["mse"] = torch.mean(torch.square((hires - fake))).numpy()
        n, c, _, _ = hires.shape
        swd = []
        for i in range(n):
            for j in range(c):
                swd.append(
                    sliced_wasserstein_cuda(hires[i, j, :, :], fake[i, j, :, :])
                )
        validation_metrics["swd"] = np.mean(swd)
        return validation_metrics

    def save_test_samples(self, alpha):
        with torch.no_grad():
            hires, lores, fake = self.gen_test_samples(
                self.gen_ema, alpha, ncand=8, return_original_range=True
            )
        tool.matplotlib_plot(
            hires,
            lores,
            fake,
            self.log_folder + '/eval/testImages_scaleX' + str(self.curr_scale) + '.png',
            data_type=self.data_type
        )

    def gen_test_samples(
        self,
        model,
        alpha,
        ncand: int = 8,
        return_original_range: bool = False,
        device=torch.device("cpu"),
    ):
        assert ncand > 0
        sample_indices = (
            torch.randint(0, len(self.dataset_test), (self.batch_size,))
            if self.sample_indices == None
            else self.sample_indices
        )
        samples_list = [self.dataset_test[i] for i in sample_indices]
        hires = torch.stack(samples_list).to(device)
        lores = tool.upscale(hires, self.max_scale)

        n, _, h, w = lores.shape
        eps_size = [n, self.noise_dim, h, w]

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
                        self.curr_scale,
                        alpha,
                    )
                    for y in range(ncand)
                ],
                dim=3,
            )

        if return_original_range:
            # stretch to origianl data range
            hires = tool.normalize(hires, self.entire_mean, self.entire_std, True)
            lores = tool.normalize(lores, self.entire_mean, self.entire_std, True)
            fake = tool.normalize(fake, self.entire_mean, self.entire_std, True)

        # downscale for visualization
        lores = tool.downscale(lores, self.max_scale)
        fake = tool.downscale(fake, self.max_scale // self.curr_scale)
        return hires, lores, fake
