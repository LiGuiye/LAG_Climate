import math

import numpy as np
import torch
from torch import nn
from utils.utils import downscale, upscale


def kaiming_scale(x, activation_slope):
    size = x.weight.size()  # (out_channel, in_channel, kernel_size, kernel_size)
    fanin = np.prod(size[1:])  # in_channel * kernel_size * kernel_size
    return math.sqrt(2.0 / ((1 + activation_slope**2) * fanin))


class ConstrainedLayer(nn.Module):
    """
    1. initialize one layer's bias to zero
    2. apply He's initialization at runtime
    """

    def __init__(self, module, equalized=True, initBiasToZero=True, activation=None):
        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized
        if activation == 'relu':
            activation_slope = 0
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            activation_slope = 0.2
            self.activation = nn.LeakyReLU(negative_slope=activation_slope)
        else:
            activation_slope = 1
            self.activation = None

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0.0, 1.0)
            # with activation slope (same as LAG TF code)
            self.scale = kaiming_scale(self.module, activation_slope)

    def forward(self, x):
        x = self.module(x * self.scale)

        if self.activation is None:
            return x
        else:
            return self.activation(x)


class ScaledConv2dWithAct(ConstrainedLayer):
    def __init__(
        self,
        nChannelsPrevious,
        nChannels,
        kernelSize=3,
        padding="same",
        stride=1,
        bias=True,
        **kwargs
    ):
        """
        A nn.Conv2d module with specific constraints
        """

        ConstrainedLayer.__init__(
            self,
            nn.Conv2d(
                nChannelsPrevious,
                nChannels,
                kernel_size=kernelSize,
                padding=padding,
                bias=bias,
                stride=stride,
            ),
            **kwargs
        )


class ResidualBlock(nn.Module):
    """
    EDSR residual blocks in Generator
    """

    def __init__(
        self, num_filters, activation='relu', conv=ScaledConv2dWithAct, block_scaling=1
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv(num_filters, num_filters, 3, activation=activation)
        self.conv2 = conv(num_filters, num_filters, 3, activation=None)
        self.first = nn.Sequential(self.conv1, self.conv2)
        self.block_scaling = block_scaling

    def forward(self, y):

        dy = self.first(y) * self.block_scaling

        return y + dy


class DownscaleBlock(nn.Module):
    def __init__(self, c_in, c_out, conv=ScaledConv2dWithAct):
        super(DownscaleBlock, self).__init__()
        conv1 = conv(c_in, c_out, 3, activation='lrelu')
        conv2 = conv(c_out, c_out, 3, activation='lrelu')
        self.conv = nn.Sequential(conv1, conv2)

    def forward(self, input):
        output = self.conv(input)
        return output


class Generator(nn.Module):
    def __init__(
        self,
        channels: int = 2,
        noise_dim: int = 64,
        residual_blocks: int = 8,
        kernelSize: int = 1, # to IMG
        residual_scaling: float = 0.0
    ):
        super().__init__()
        # input dimension (image + eps)
        conv = ScaledConv2dWithAct

        # 1 conv and 8 residual block as fixed beginning -------------
        self.input_layer = nn.Sequential(
            conv(channels + noise_dim, 256, 3),
            *[
                ResidualBlock(256, block_scaling=residual_scaling)
                for _ in range(residual_blocks)
            ]
        )

        # changing layers depends on different stage -------------

        # upsample block
        self.downscale = nn.ModuleList(
            [
                DownscaleBlock(256, 128),
                DownscaleBlock(128, 64),
                DownscaleBlock(64, 64),
                DownscaleBlock(64, 64),
                DownscaleBlock(64, 64),
                DownscaleBlock(64, 64)
            ]
        )

        # to_Img
        self.toImg = nn.ModuleList(
            [
                conv(256, channels, kernelSize),
                conv(128, channels, kernelSize),
                conv(64, channels, kernelSize),
                conv(64, channels, kernelSize),
                conv(64, channels, kernelSize),
                conv(64, channels, kernelSize),
                conv(64, channels, kernelSize)
            ]
        )

    def forward(self, input, eps, scale_factor, alpha):
        image = self.input_layer(torch.cat([input, eps], dim=1))

        imgs = [image]
        log_scale = scale_factor.bit_length() - 1 # int(log2)
        for i in range(log_scale):
            imgs.append(self.downscale[i](downscale(imgs[i])))

        if 0 <= alpha < 1:
            # transition stage
            out = (1 - alpha) * downscale(self.toImg[i](imgs[-2])) + alpha * self.toImg[i+1](imgs[-1])
        else:
            # training stage
            out = self.toImg[i+1](imgs[-1])
        return out


class SpaceToChannel(nn.Module):
    def __init__(self, n=2):
        super(SpaceToChannel, self).__init__()
        self.n = n

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(-1, c, h // self.n, self.n, w // self.n, self.n)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(-1, c * (self.n**2), h // self.n, w // self.n)
        return x


class UpscaleBlock(nn.Module):
    def __init__(self, c_in, c_out, conv=ScaledConv2dWithAct):
        super(UpscaleBlock, self).__init__()
        conv1 = conv(c_in, c_in, 3, activation='lrelu')
        conv2 = conv(4 * c_in, c_out, 3, activation='lrelu')
        self.conv = nn.Sequential(conv1, SpaceToChannel(), conv2)

    def forward(self, input):
        output = self.conv(input)
        return output


class Discriminator(nn.Module):
    def __init__(
        self,
        channels: int = 2,
        residual_blocks: int = 8,
        kernelSize: int = 1, # from IMG
    ):
        super().__init__()
        conv = ScaledConv2dWithAct

        # changing layers depends on different stage -------------
        self.fromImg = nn.ModuleList(
            [
                conv(channels, 64, kernelSize, activation='lrelu'), # x64
                conv(channels, 64, kernelSize, activation='lrelu'), # x32
                conv(channels, 64, kernelSize, activation='lrelu'), # x16
                conv(channels, 64, kernelSize, activation='lrelu'), # x8
                conv(channels, 64, kernelSize, activation='lrelu'), # x4
                conv(channels, 128, kernelSize, activation='lrelu'), # x2
                conv(channels, 256, kernelSize, activation='lrelu') # x2 transition
            ]
        )

        # upsample block
        self.upscale = nn.ModuleList(
            [
                UpscaleBlock(64, 64),  # x64
                UpscaleBlock(64, 64),  # x32
                UpscaleBlock(64, 64),  # x16
                UpscaleBlock(64, 64),  # x8
                UpscaleBlock(64, 128),  # x4
                UpscaleBlock(128, 256),  # x2
            ]
        )

        # fixed last block -------------
        self.last_block = nn.Sequential(
            *[
                conv(
                    256 + (channels if block == 0 else 0),
                    256,
                    3,
                    activation='lrelu',
                )
                for block in range(residual_blocks)
            ]
        )

        center = torch.ones((1, 256, 1, 1))
        center[:, ::2, :, :] = -1
        self.register_buffer('center', center)

        max_scale_factor = 64
        self.n_layer = max_scale_factor.bit_length() - 1

    def forward(self, x0_input, lowres_x_delta, scale_factor, alpha):
        x0 = x0_input

        log_scale = scale_factor.bit_length() - 1 # int(log2)
        for i in reversed(range(log_scale)):
            idx = self.n_layer - i - 1

            if i == log_scale - 1:
                # only beginning layer
                out = self.fromImg[idx](x0)
            out = self.upscale[idx](out)

            if i == log_scale - 1 and 0 <= alpha < 1:
                # only at transition stage and top layer block
                skip_rgb = upscale(x0)
                skip_rgb = self.fromImg[idx + 1](skip_rgb)
                out = (1 - alpha) * skip_rgb + alpha * out

        out = self.last_block(torch.cat([out, lowres_x_delta], dim=1))
        return out * self.center
