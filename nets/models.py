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
        image_channel: int = 2,
        noise_dim: int = 64,
        residual_blocks: int = 8,
        kernelSize_toRGB: int = 1,
        residual_scaling = 0
    ):
        super().__init__()
        # input dimension (image + eps)
        c_in = image_channel + noise_dim
        self.conv = ScaledConv2dWithAct

        # 1 conv and 8 residual block as fixed beginning -------------
        self.input_layer = nn.Sequential(
            self.conv(c_in, 256, 3, activation=None),
            *[
                ResidualBlock(256, conv=self.conv, block_scaling=residual_scaling)
                for _ in range(residual_blocks)
            ]
        )

        # changing layers depends on different stage -------------
        # upsample block
        self.downscale_x2 = DownscaleBlock(256, 128)
        self.downscale_x4 = DownscaleBlock(128, 64)
        self.downscale_x8 = DownscaleBlock(64, 64)
        self.downscale_x16 = DownscaleBlock(64, 64)
        self.downscale_x32 = DownscaleBlock(64, 64)
        self.downscale_x64 = DownscaleBlock(64, 64)
        # to_rgb
        self.to_rgb = self.conv(256, image_channel, kernelSize_toRGB, activation=None)
        self.to_rgb_scale2 = self.conv(
            128, image_channel, kernelSize_toRGB, activation=None
        )
        self.to_rgb_scale4 = self.conv(
            64, image_channel, kernelSize_toRGB, activation=None
        )
        self.to_rgb_scale8 = self.conv(
            64, image_channel, kernelSize_toRGB, activation=None
        )
        self.to_rgb_scale16 = self.conv(
            64, image_channel, kernelSize_toRGB, activation=None
        )
        self.to_rgb_scale32 = self.conv(
            64, image_channel, kernelSize_toRGB, activation=None
        )
        self.to_rgb_scale64 = self.conv(
            64, image_channel, kernelSize_toRGB, activation=None
        )

        self.max_scale_factor = 64

    def progress(self, feat, module):
        out = module(downscale(feat))
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            # transition stage
            skip_rgb = downscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            # training stage
            out = module2(feat2)
        return out

    def forward(self, input, eps, scale_factor, alpha):
        image = self.input_layer(torch.cat([input, eps], dim=1))

        image_scale2 = self.progress(image, self.downscale_x2)
        if scale_factor == 2:
            out = self.output(
                image, image_scale2, self.to_rgb, self.to_rgb_scale2, alpha
            )
            return out

        image_scale4 = self.progress(image_scale2, self.downscale_x4)
        if scale_factor == 4:
            out = self.output(
                image_scale2,
                image_scale4,
                self.to_rgb_scale2,
                self.to_rgb_scale4,
                alpha,
            )
            return out

        image_scale8 = self.progress(image_scale4, self.downscale_x8)
        if scale_factor == 8:
            out = self.output(
                    image_scale4,
                    image_scale8,
                    self.to_rgb_scale4,
                    self.to_rgb_scale8,
                    alpha,
                )
            return out

        image_scale16 = self.progress(image_scale8, self.downscale_x16)
        if scale_factor == 16:
            out = self.output(
                image_scale8,
                image_scale16,
                self.to_rgb_scale8,
                self.to_rgb_scale16,
                alpha,
            )
            return out

        image_scale32 = self.progress(image_scale16, self.downscale_x32)
        if scale_factor == 32:
            out = self.output(
                image_scale16,
                image_scale32,
                self.to_rgb_scale16,
                self.to_rgb_scale32,
                alpha,
            )
            return out

        image_scale64 = self.progress(image_scale32, self.downscale_x64)
        if scale_factor == 64:
            out = self.output(
                image_scale32,
                image_scale64,
                self.to_rgb_scale32,
                self.to_rgb_scale64,
                alpha,
            )
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
        image_channel: int = 2,
        residual_blocks: int = 8,
        kernelSize_fromRGB: int = 1,
    ):
        super().__init__()

        self.conv = ScaledConv2dWithAct
        # changing layers depends on different stage -------------
        self.from_rgb = nn.ModuleList(
            [
                self.conv(
                    image_channel, 64, kernelSize_fromRGB, activation='lrelu'
                ),  # x64
                self.conv(
                    image_channel, 64, kernelSize_fromRGB, activation='lrelu'
                ),  # x32
                self.conv(
                    image_channel, 64, kernelSize_fromRGB, activation='lrelu'
                ),  # x16
                self.conv(
                    image_channel, 64, kernelSize_fromRGB, activation='lrelu'
                ),  # x8
                self.conv(
                    image_channel, 64, kernelSize_fromRGB, activation='lrelu'
                ),  # x4
                self.conv(
                    image_channel, 128, kernelSize_fromRGB, activation='lrelu'
                ),  # x2
                self.conv(
                    image_channel, 256, kernelSize_fromRGB, activation='lrelu'
                ),  # x2 transition
            ]
        )

        # upsample block
        self.upscale_block = nn.ModuleList(
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
                self.conv(
                    256 + (image_channel if block == 0 else 0),
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

        self.max_scale_factor = 64
        self.n_layer = int(np.log2(self.max_scale_factor))

    def forward(self, x0_input, lowres_x_delta, scale_factor, alpha):
        x0 = x0_input

        log_scale = int(np.log2(scale_factor))
        for i in reversed(range(log_scale)):
            index = self.n_layer - i - 1

            if i == log_scale - 1:
                # only beginning layer
                out = self.from_rgb[index](x0)
            out = self.upscale_block[index](out)

            if i == log_scale - 1 and 0 <= alpha < 1:
                # only at transition stage and top layer block
                skip_rgb = upscale(x0)
                skip_rgb = self.from_rgb[index + 1](skip_rgb)
                out = (1 - alpha) * skip_rgb + alpha * out


        out = self.last_block(torch.cat([out, lowres_x_delta], dim=1))
        return out * self.center
