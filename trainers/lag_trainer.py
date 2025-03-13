import os
from glob import glob

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import *
from utils.metrics import variogram, calc_metrics
from nets.dataset import get_dataset
from nets.models import Generator, Discriminator
from data.config import get_dataset_path


class LAGTrainer:
    def __init__(self, args):
        super(LAGTrainer, self).__init__()
        self.report_step = args.report_step
        self.log_folder = f"results/{args.expid}"
        self.device, self.gpu_num, self.num_workers = check_cuda()
        # transition epochs:  args.epoch[0]
        # stabilization epochs:  args.epoch[1]
        epoch = [int(i) for i in args.epoch.split(',')]
        self.transition_epoch, self.stablization_epoch = epoch
        self.epochs_per_stage = sum(epoch)
        self.load_data(args)
        self.load_model(args)

    def load_data(self, args):
        self.channel = args.channel
        self.data_name = args.data_name
        self.batch_size = args.batch_size
        (path_train, path_test, train_mean, train_std) = get_dataset_path(self.data_name)
        self.train_mean = [float(i) for i in train_mean.split(',')]
        self.train_std = [float(i) for i in train_std.split(',')]
        self.dataset_train = get_dataset(
            path_train,
            args.data_size,
            self.train_mean,
            self.train_std
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
            self.train_mean,
            self.train_std
        )

    def create_folder(self):
        mkdir(
            [
                self.log_folder,
                self.log_folder + '/checkpoint',
                self.log_folder + '/eval',
                self.log_folder + '/args'
            ]
        )

    def load_model(self, args):
        self.max_scale = args.max_scale
        self.noise_dim = args.noise_dim
        self.n_critic = args.n_critic
        self.vario_weight = args.vario_weight
        self.mass_weight = args.mass_weight
        self.center_weight = args.center_weight
        self.curr_scale = self.max_scale if args.memtest else 2
        self.gen = Generator(
            args.channel,
            args.noise_dim,
            kernelSize=args.kernelSize,
            residual_blocks=args.residual_blocks,
            residual_scaling=args.residual_scaling
        )
        self.gen_ema = build_ema(self.gen)
        self.dis = Discriminator(
            args.channel,
            kernelSize=args.kernelSize,
            residual_blocks=args.residual_blocks
        )

        # optimizer
        self.g_optim = optim.Adam(
            get_model(self.gen).parameters(),
            lr=args.lr,
            betas=(0.0, 0.99)
        )
        self.d_optim = optim.Adam(
            get_model(self.dis).parameters(),
            lr=args.lr,
            betas=(0.0, 0.99)
        )

        if args.reset:
            print("Training from scratch!")
            remove_all(self.log_folder)
            self.create_folder()
            save_args(args, f"{self.log_folder}/args")
            self.curr_epoch = 0
        else:
            # try to load previous weights
            if os.path.exists(f"{self.log_folder}/checkpoint/tmp"):
                ckpt_path = glob(f"{self.log_folder}/checkpoint/tmp/*.pt")[0]
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.curr_epoch = int(ckpt_path.split('.')[0].split('_')[-1])
                self.curr_scale = int(ckpt_path.split('.')[0].split('_')[-3])
            else:
                ckpt_path = f"{self.log_folder}/checkpoint"
                scale_max = max(
                    [
                        int(i.split('.')[0].split('_')[-1])
                        for i in glob(f"{ckpt_path}/*.pt")
                    ]
                )
                assert scale_max < self.max_scale, 'Model have already been trained'
                ckpt = torch.load(
                    f"{ckpt_path}/LAG_scale_{scale_max}.pt", map_location=self.device)
                self.curr_epoch = 0
                self.curr_scale = scale_max * 2

            print("Training continued!")

            self.gen.load_state_dict(ckpt['gen_state_dict'])
            self.gen_ema.load_state_dict(ckpt['gen_ema_state_dict'])
            self.dis.load_state_dict(ckpt['dis_state_dict'])
            self.g_optim.load_state_dict(ckpt['optimizer_gen_state_dict'])
            self.d_optim.load_state_dict(ckpt['optimizer_dis_state_dict'])
            move_optim(self.g_optim, self.device)
            move_optim(self.d_optim, self.device)

        # move model to device
        self.gen = torch.nn.DataParallel(self.gen).to(self.device)
        self.dis = torch.nn.DataParallel(self.dis).to(self.device)

        print_args(args)

    def calc_gp(self, n, real, fake_eps, lores, alpha):
        """
        a gradient norm penalty to achieve Lipschitz continuity.
        """
        gp_alpha = torch.rand(n, 1, 1, 1, device=self.device)
        mixed = gp_alpha * real.data + (1 - gp_alpha) * fake_eps.data
        mixed.requires_grad = True

        # downscale to largest image resolution
        mixed = downscale(mixed, self.remaining_scale)
        # upscale to lores and calc residual
        mixed_residual = torch.abs(lores - upscale(mixed, self.max_scale))
        # upscale to current target image resolution
        mixdown = upscale(mixed, self.remaining_scale)
        mixed_scores = torch.mean(self.dis(mixdown, mixed_residual, self.curr_scale, alpha), 1).sum()
        grad_mixed = torch.autograd.grad(outputs=mixed_scores, inputs=mixed, create_graph=True)[0]
        grad_norm = grad_mixed.view(grad_mixed.size(0), -1).norm(2, dim=1)

        loss_gp = 10 * ((grad_norm - 1) ** 2).mean()
        return loss_gp

    def train_epoch(self):
        start_time = get_time()
        curr_stage_time = None

        logger = get_logger(f"{self.log_folder}/checkpoint/log.txt")

        total_epochs = ilog2(self.max_scale) * self.epochs_per_stage
        trained_epochs = (ilog2(self.curr_scale) - 1) * self.epochs_per_stage + self.curr_epoch
        remain_epochs = total_epochs - trained_epochs

        iters_per_epoch = self.dataset_length // self.batch_size
        iters_per_transition = self.transition_epoch * iters_per_epoch

        if self.curr_epoch == 0:
            alpha = 0
        else:
            iteration = self.curr_epoch * iters_per_epoch + 1
            alpha = min(1, (1 / iters_per_transition) * iteration)

        for epoch in range(remain_epochs):
            if self.curr_epoch == 0 or epoch == 0:
                logger.info(
                    f"Scale factor: {self.curr_scale} / Total epochs remain: {total_epochs - trained_epochs}")
                logger.info('Transition stage start:')
            if self.curr_epoch == 0:
                alpha = 0
            elif self.curr_epoch == self.transition_epoch:
                logger.info('Stabilization stage start:')

            self.remaining_scale = self.max_scale // self.curr_scale
            for i, x in enumerate(self.loader_train):

                x = x.to(self.device)
                lores = upscale(x, self.max_scale)
                real = upscale(x, self.remaining_scale)

                # extra data process during transition stage to stabilize the training
                # 1. `downscale(upscale(real))` to remove detailed information
                # 2. gradually blend them together (alpha gradually increase from 0 to 1)
                real = (1 - alpha) * downscale(upscale(real)) + alpha * real

                # 1. train Discriminator
                self.dis.zero_grad()
                self.d_optim.zero_grad()

                b, _, h, w = lores.shape
                eps = torch.randn(b, self.noise_dim, h, w, device=self.device)
                fake_eps = self.gen(lores, eps, self.curr_scale, alpha)
                lores_fake_eps = upscale(downscale(fake_eps, self.remaining_scale), self.max_scale)

                # C(x,y)
                latent_real = self.dis(real, torch.abs(lores-lores), self.curr_scale, alpha)
                # C(G(z,y),y)
                latent_fake_eps = self.dis(fake_eps, torch.abs(lores-lores_fake_eps), self.curr_scale, alpha)

                # WGAN-GP loss
                loss_dreal = -torch.mean(latent_real)
                loss_dfake = torch.mean(latent_fake_eps)
                loss_gp = self.calc_gp(b, real, fake_eps, lores, alpha)
                loss_disc = loss_dfake + loss_dreal + loss_gp

                loss_disc.backward()
                self.d_optim.step()

                # 2. train Generator
                if (i + 1) % self.n_critic == 0:
                    self.gen.zero_grad()
                    self.g_optim.zero_grad()

                    eps = torch.randn(b, self.noise_dim, h, w, device=self.device)
                    fake_eps = self.gen(lores, eps, self.curr_scale, alpha)
                    fake = self.gen(lores, torch.zeros_like(eps, device=self.device), self.curr_scale, alpha)

                    lores_fake_eps = upscale(
                        downscale(fake_eps, self.remaining_scale), self.max_scale)
                    lores_fake = upscale(
                        downscale(fake, self.remaining_scale), self.max_scale)

                    # WGAN-GP loss -------------
                    # P(G(y, z), y)
                    latent_fake_eps = self.dis(fake_eps, torch.abs(lores-lores_fake_eps), self.curr_scale, alpha)
                    loss_gfake = -torch.mean(latent_fake_eps)

                    # L_center -------------
                    # P(x, y)
                    latent_real = self.dis(real, torch.abs(lores-lores), self.curr_scale, alpha)
                    # P(G(y, 0), y)
                    latent_fake = self.dis(fake, torch.abs(lores-lores_fake), self.curr_scale, alpha)
                    loss_gmse = self.center_weight * F.mse_loss(latent_real, latent_fake)

                    # variogram -------------
                    loss_gvario = self.vario_weight * variogram(real, fake, self.device) if self.vario_weight else 0
                    loss_gmass = self.mass_weight * F.mse_loss(lores, upscale(fake, self.curr_scale)) if self.mass_weight else 0

                    # Total G loss -------------
                    loss_gen = loss_gfake + loss_gmse + loss_gvario + loss_gmass

                    loss_gen.backward()
                    self.g_optim.step()
                    update_ema(self.gen_ema, self.gen)

                # log
                trained_step = trained_epochs * iters_per_epoch + i + 1
                if (trained_step % self.report_step) == 0 and ((i + 1) >= self.n_critic):
                    hires, _, fake = self.gen_test_samples(self.gen_ema, alpha, ncand=1)
                    val_mse, val_swd = calc_metrics(hires, fake, ["swd", "mse"]).values()

                    logger.info(f"{self.curr_epoch+1}/{self.epochs_per_stage} | "+
                                f"Loss_G: wgan {rd(loss_gfake)}, center {rd(loss_gmse)}, vario {rd(loss_gvario)}, mass {rd(loss_gmass)}, total {rd(loss_gen)} | "+
                                f"Loss_D: wgan {rd(loss_dreal+loss_dfake)}, wgan-gp {rd(loss_gp)}, total {rd(loss_disc)} | "+
                                f"Validation: MSE {rd(val_mse)} SWD {rd(val_swd)}")

                # update alpha and iteration count
                iteration = self.curr_epoch * iters_per_epoch + i + 1
                alpha = min(1, (1 / iters_per_transition) * iteration)

            trained_epochs += 1
            self.curr_epoch += 1
            if self.curr_epoch == self.epochs_per_stage:
                self.gen_test_samples(self.gen_ema, alpha, savePath=f"{self.log_folder}/eval/testImages_{self.curr_scale}X.png")
                self.save_model()

                finish_time = get_time() - start_time if curr_stage_time is None else get_time() - curr_stage_time
                curr_stage_time = get_time()
                logger.info(
                    f"Current scale finished. Took {rd(finish_time/3600)} hours to complete.")
                # new transition and training stages begin
                self.curr_scale *= 2
                self.curr_epoch = 0
            else:
                self.save_model(self.curr_epoch)

        # print and save total time consuming
        finish_time = get_time() - start_time
        logger.info(
            f"Training finished. Took {rd(finish_time/3600)} hours to complete.")

    def save_model(self, curr_epoch=None):
        if curr_epoch is None:
            save_path = f"{self.log_folder}/checkpoint/LAG_scale_{self.curr_scale}.pt"
            remove_all(self.log_folder + '/checkpoint/tmp')
        else:
            remove_all(self.log_folder + '/checkpoint/tmp')
            mkdir([self.log_folder + '/checkpoint/tmp'])
            save_path = (
                self.log_folder
                + '/checkpoint/tmp/LAG_scale_'
                + str(self.curr_scale)
                + '_epoch_'
                + str(curr_epoch)
                + '.pt'
            )
        save_weights = {
            'gen_state_dict': get_model(self.gen).state_dict(),
            'gen_ema_state_dict': get_model(self.gen_ema).state_dict(),
            'dis_state_dict': get_model(self.dis).state_dict(),
            'optimizer_gen_state_dict': self.g_optim.state_dict(),
            'optimizer_dis_state_dict': self.d_optim.state_dict(),
        }
        torch.save(save_weights, save_path)

    def gen_test_samples(
        self,
        model,
        alpha,
        ncand: int = 8,
        back2raw: bool = False,
        savePath=None
    ):
        assert ncand > 0
        sample_indices = [0]
        hires = torch.stack([self.dataset_test[i] for i in sample_indices]).to(self.device)
        lores = upscale(hires, self.max_scale)
        hires = upscale(hires, self.remaining_scale)

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
                            device=self.device
                        ),
                        self.curr_scale,
                        alpha,
                    )
                    for y in range(ncand)
                ],
                dim=3,
            )

        if back2raw:
            # stretch to origianl data range
            hires = normalize(hires, self.train_mean, self.train_std, True)
            lores = normalize(lores, self.train_mean, self.train_std, True)
            fake = normalize(fake, self.train_mean, self.train_std, True)

        # downscale for visualization
        lores = downscale(lores, self.max_scale)
        fake = downscale(fake, self.remaining_scale)
        hires = downscale(hires, self.remaining_scale)

        hires, lores, fake = hires.cpu(), lores.cpu(), fake.cpu()
        if not savePath is None:
            save_plot(hires, lores, fake, savePath, self.data_name)
        return hires, lores, fake
