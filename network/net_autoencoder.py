# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse
# @File : net_autoencoder.py
# @Time : 2021/11/9 13:18
import torch
import torch.nn as nn
import numpy as np

from network.loss import mse_loss, ssim_loss, nucnorm_loss
from tools import utils
from args_auto import Args as args

EXP = 1e-6


# Convolution operation
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# dense conv
class Dense_ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = torch.cat((x, out), 1)
        return out


# dense block
class Dense_Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size, stride, dense_out):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.add_module('dense_conv' + str(i),
                            Dense_ConvLayer(in_channels + i * out_channels, out_channels, kernel_size, stride))
        self.adjust_conv = ConvLayer(in_channels + num_layers * out_channels, dense_out, kernel_size, stride)

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            # print('num_block - ' + str(i))
            dense_conv = getattr(self, 'dense_conv' + str(i))
            out = dense_conv(out)
        out = self.adjust_conv(out)
        return out


# encoder network, extract features
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels, num_layers, dense_out, part_out):
        super().__init__()
        # 4 pooling, 256 -->> 16
        self.kernel_size = 3
        self.stride = 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = ConvLayer(in_channels, out_channels1, self.kernel_size, self.stride)
        self.dense_blocks = nn.Sequential(
            Dense_Block(num_layers, out_channels1, out_channels, self.kernel_size, self.stride, dense_out),
            nn.MaxPool2d(2, 2),
            Dense_Block(num_layers, dense_out, out_channels, self.kernel_size, self.stride, dense_out),
            nn.MaxPool2d(2, 2),
            # Dense_Block(num_layers, dense_out, out_channels, self.kernel_size, self.stride, dense_out),
            # nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.pool(out1)
        # out = out1
        out = self.dense_blocks(out)
        # out_ad = self.adjust_conv(out)  # 1*1
        # c_out = self.commn_conv(out)
        # s_out = out - c_out
        return out1, out


# decoder network for recons loss
class Decoder_rec(nn.Module):
    def __init__(self, in_channels, out_channels, train_flag=False):
        super().__init__()
        self.kernel_size = 3
        self.stride = 1
        self.train_flag = train_flag
        self.up = nn.Upsample(scale_factor=2)
        self.shape_adjust = utils.UpsampleReshape_eval()
        self.conv1 = ConvLayer(in_channels, int(in_channels / 2), self.kernel_size, self.stride)
        self.conv_block = nn.Sequential(
            ConvLayer(int(in_channels / 2), int(in_channels / 2), self.kernel_size, self.stride),
            nn.Upsample(scale_factor=2),
            ConvLayer(int(in_channels / 2), int(in_channels / 4), self.kernel_size, self.stride),
            nn.Upsample(scale_factor=2),
            # ConvLayer(int(in_channels / 4), int(in_channels / 8), self.kernel_size, self.stride),
            # nn.Upsample(scale_factor=2),
            # ConvLayer(int(in_channels / 4), out_channels, self.kernel_size, self.stride)
        )
        self.conv_last = ConvLayer(int(in_channels / 4), out_channels, 1, self.stride)
        # self.conv_last1 = ConvLayer(out_channels, out_channels, 1, self.stride)

        # self.act_func = nn.Sigmoid()

    def forward(self, c1, x1):
        w = [1.0, 1.0]
        # x1 = x1 + w[0] * c_ad
        out = self.up(self.conv1(x1))
        # out = self.conv1(x1)
        # out = out + w[0] * c_ad
        out = self.conv_block(out)

        if not self.train_flag:
            out = self.shape_adjust(c1, out)
        out = out + w[1] * c1
        out = self.conv_last(out)
        # out = self.conv_last1(out)
        # out = self.act_func(out)
        return out


class Auto_Encoder_single(nn.Module):
    def __init__(self, in_channels, out_channels, en_out_channels1, en_out_channels, num_layers, dense_out, part_out, train_flag=True):
        super().__init__()
        self.fea_encoder1 = Encoder(in_channels, en_out_channels1, en_out_channels, num_layers, dense_out, part_out)
        self.recon_decoder1 = Decoder_rec(part_out, out_channels, train_flag=train_flag)

    def forward(self, x1):
        x1_norm = x1 / 255
        c_sh, c_de = self.fea_encoder1(x1_norm)
        return c_sh, c_de

    def reconsturce(self, x1):
        x1_norm = x1 / 255
        # -------------------------------------
        c1, c_ad, x1_s, x1_c = self.fea_encoder1(x1_norm)
        # reconstruct
        out1 = self.recon_decoder1(c1, c_ad)
        # -------------------------------------
        # out1 = tools.normalize_tensor(out1)
        out1 = out1 * 255
        # -------------------------------------
        outputs = {'salient1': x1_s,
                   'common1': x1_c,
                   'recons1': out1
                   }
        return outputs

    def train_module(self, x1):
        x1_norm = x1 / 255
        c1, c_ad = self.fea_encoder1(x1_norm)

        # reconstruct
        out1 = self.recon_decoder1(c1, c_ad)
        # -------------------------------------
        out1 = out1 * 255
        # -------------------------------------
        loss_recon1 = mse_loss(out1, x1)
        loss_ssim1 = 1 - ssim_loss(out1, x1, normalize=True)

        w = args.w
        total_loss = w[0] * loss_recon1 + \
                     w[1] * loss_ssim1

        outputs = {'out': out1,
                   'recon_loss': w[0] * loss_recon1,
                   'ssim_loss': w[1] * loss_ssim1,
                   'total_loss': total_loss
                   }
        return outputs
