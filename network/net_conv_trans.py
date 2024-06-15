# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse
# @File : net_transf.py
# @Time : 2021/11/8 16:16

import torch
import torch.nn as nn
import numpy as np

from network.transformer_cam import cross_encoder, PatchEmbed_tensor, Recons_tensor
from network.loss import mse_loss, ssim_loss, feature_loss
from tools import utils
# from fuse_network.net_autoencoder import Encoder


EPSILON = 1e-6


def remove_mean(x):
    (b, ch, h, w) = x.size()

    tensor = x.reshape(b, ch, h * w)
    t_mean = torch.mean(tensor, 2)

    t_mean = t_mean.view(b, ch, 1, 1)
    t_mean = t_mean.repeat(1, 1, h, w)

    out = x - t_mean
    return out, t_mean


class Weight(nn.Module):
    def __init__(self, ch_s, ks_s, ch_d, ks_d):
        super(Weight, self).__init__()
        # weight for features
        # ks_s = 17
        weight_sh = torch.ones([1, 1, ks_s, ks_s])
        reflection_padding_sh = int(np.floor(ks_s / 2))
        # weight_sh = torch.from_numpy(np.array([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]))
        self.conv_sh = torch.nn.Conv2d(ch_s, ch_s, (ks_s, ks_s), stride=1, padding=reflection_padding_sh, bias=False)
        self.conv_sh.weight.data = (1 / (ks_s * ks_s)) * weight_sh.repeat(ch_s, ch_s, 1, 1).float()
        self.conv_sh.requires_grad_(False)

        weight_de = torch.ones([1, 1, ks_d, ks_d])
        reflection_padding_de = int(np.floor(ks_d / 2))
        self.conv_de = torch.nn.Conv2d(ch_d, ch_d, (ks_d, ks_d), stride=1, padding=reflection_padding_de, bias=False)
        self.conv_de.weight.data = (1 / (ks_d * ks_d)) * weight_de.repeat(ch_d, ch_d, 1, 1).float()
        self.conv_de.requires_grad_(False)

    def for_sh(self, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        # g_x = x - self.conv_sh(x)
        # g_y = y - self.conv_sh(y)
        g_x = torch.sqrt((x - self.conv_sh(x)) ** 2)
        g_y = torch.sqrt((y - self.conv_sh(y)) ** 2)
        w_x = g_x / (g_x + g_y + EPSILON)
        w_y = g_y / (g_x + g_y + EPSILON)
        
        w_x = w_x.detach()
        w_y = w_y.detach()
        return w_x, w_y

    def for_de(self, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        # g_x = torch.sqrt((x - self.conv_de(x)) ** 2)
        # g_y = torch.sqrt((y - self.conv_de(y)) ** 2)
        g_x = self.conv_de(x)
        g_y = self.conv_de(y)
        w_x = g_x / (g_x + g_y + EPSILON)
        w_y = g_y / (g_x + g_y + EPSILON)
        
        w_x = w_x.detach()
        w_y = w_y.detach()
        return w_x, w_y


# Convolution operation
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# decoder network for final fusion
class Decoder_fusion(nn.Module):
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
            ConvLayer(int(in_channels / 4), int(in_channels / 4), self.kernel_size, self.stride),
            # nn.Upsample(scale_factor=2),
            # ConvLayer(int(in_channels / 4), out_channels, self.kernel_size, self.stride)
        )
        last_ch = int(in_channels / 4)
        self.conv_last = ConvLayer(last_ch, out_channels, 1, self.stride)
        # self.conv_last1 = ConvLayer(out_channels, out_channels, 1, self.stride)

        # self.act_func = nn.Sigmoid()
        ks_s = 3
        ks_d = 3
        self.weight = Weight(last_ch, ks_s, in_channels, ks_d)

    def forward(self, ir_sh, vi_sh, ir_de, vi_de, x1):
        wd = self.weight.for_de(ir_de, vi_de)  # average
        out = x1
        
        out = out + wd[0] * ir_de + wd[1] * vi_de
        out = self.up(self.conv1(out))
        out = self.conv_block(out)

        if not self.train_flag:
            out = self.shape_adjust(ir_sh, out)
        ws = self.weight.for_sh(ir_sh, vi_sh)  # gradient
        out = out + 0.5 * ws[0] * ir_sh + ws[1] * vi_sh
        out = self.conv_last(out)
        return out


class Trans_FuseNet(nn.Module):
    def __init__(self, img_size, patch_size, en_out_channels1, out_channels, part_out, train_flag, 
                 depth_self, depth_cross, n_heads, mlp_ratio, qkv_bias, p, attn_p):
        super().__init__()
        self.img_size = img_size  # 32*32
        self.patch_size = patch_size  # 2*2
        self.part_out = part_out  # 128
        self.patch_embed_tensor = PatchEmbed_tensor(img_size)
        self.recons_tensor = Recons_tensor(img_size)
        self.embed_dim = part_out * patch_size * patch_size  # 512
        self.num_patches = int(img_size / patch_size) * int(img_size / patch_size)  # 16*16
        
        self.cross_atten_block = cross_encoder(self.img_size, self.patch_size, self.embed_dim, self.num_patches, depth_self,
                                               depth_cross, n_heads, mlp_ratio, qkv_bias, p, attn_p)

        # gradient
        self.conv_gra = torch.nn.Conv2d(en_out_channels1, en_out_channels1, (3, 3), stride=1, padding=1,
                                        bias=False)
        weight = torch.from_numpy(np.array([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]))
        self.conv_gra.weight.data = weight.repeat(en_out_channels1, en_out_channels1, 1, 1).float()

        self.decoder_fusion = Decoder_fusion(part_out, out_channels, train_flag)

    def forward(self, ir_de, ir_sh, vi_de, vi_sh, shift_flag):
        # based on 32*32, for arbitrary image size
        ir_patched, patches_paddings = self.patch_embed_tensor(ir_de)
        b, c, N, h, w = ir_patched.shape
        vi_patched, _ = self.patch_embed_tensor(vi_de)
        c_f = []
        ir_self = []
        vi_self = []
        ir_roll = []
        vi_roll = []
        ir_cross = []
        vi_cross = []
        for i in range(N):
            c_f_p, ir_self_p, vi_self_p, ir_roll_p, vi_roll_p, vi_cross_p, ir_cross_p = \
                self.cross_atten_block(ir_patched[:, :, i, :, :], vi_patched[:, :, i, :, :], shift_flag)
            ir_self.append(ir_self_p)
            vi_self.append(vi_self_p)
            ir_roll.append(ir_roll_p)
            vi_roll.append(vi_roll_p)
            ir_cross.append(ir_cross_p)
            vi_cross.append(vi_cross_p)
            c_f.append(c_f_p)
        if b == 1:
            ir_self = torch.cat(ir_self, dim=0).unsqueeze(dim=1)
            vi_self = torch.cat(vi_self, dim=0).unsqueeze(dim=1)
            ir_roll = torch.cat(ir_roll, dim=0).unsqueeze(dim=1)
            vi_roll = torch.cat(vi_roll, dim=0).unsqueeze(dim=1)
            ir_cross = torch.cat(ir_cross, dim=0).unsqueeze(dim=1)
            vi_cross = torch.cat(vi_cross, dim=0).unsqueeze(dim=1)
            c_f = torch.cat(c_f, dim=0).unsqueeze(dim=1)
        else:
            ir_self = torch.cat(ir_self, dim=0)
            vi_self = torch.cat(vi_self, dim=0)
            ir_roll = torch.cat(ir_roll, dim=0)
            vi_roll = torch.cat(vi_roll, dim=0)
            ir_cross = torch.cat(ir_cross, dim=0)
            vi_cross = torch.cat(vi_cross, dim=0)
            c_f = torch.cat(c_f, dim=0)
        
        c_f = c_f.permute(1, 2, 0, 3, 4)  # b, c, N, h, w
        c_f = self.recons_tensor(c_f, patches_paddings)
        # -----------------------------------
        in_put = c_f
        # -----------------------------------
        out = self.decoder_fusion(ir_sh, vi_sh, ir_de, vi_de, in_put)
        out = utils.normalize_tensor(out)
        out = out * 255
        # -----------------------------------
        outputs = {'out': out}

        return outputs

    # training phase
    def train_module(self, x_ir, x_vi, ir_sh, vi_sh, ir_de, vi_de, shift_flag, gra_loss, order_loss):
        # cross attention module
        c_f, ir_self, vi_self, ir_roll, vi_roll, vi_cross, ir_cross = self.cross_atten_block(ir_de, vi_de, shift_flag)
        in_put = c_f
        # -----------------------------------
        # visiualize the middle features
        ir_s = utils.recons_midle_feature(ir_self)
        vi_s = utils.recons_midle_feature(vi_self)
        vi_c = utils.recons_midle_feature(vi_cross)
        ir_c = utils.recons_midle_feature(ir_cross)
        
        fuse = utils.recons_midle_feature(in_put)
        middle_temp = [ir_s, vi_s, ir_c, vi_c, fuse]
        # -----------------------------------
        # decoder fusion
        out = self.decoder_fusion(ir_sh, vi_sh, ir_de, vi_de, in_put)
        out = utils.normalize_tensor(out)
        out = out * 255
        # -----------------------------------
        loss_pix, temp = order_loss(out, x_ir, x_vi)
  
        loss_gra, gp, gxir, gxvi, g_target = gra_loss(out, x_ir, x_vi)
        weight = [gxir, gxvi, g_target, gp, temp]
        # loss_mean = mean_loss(out, x_vi, x_ir)
        loss_mean = 0.0

        # loss_sh, loss_mi, loss_de = feature_loss(vgg, x_ir, x_vi, out)
        # loss_fea = loss_sh + loss_mi + loss_de
        loss_sh, loss_mi, loss_de = 0.0, 0.0, 0.0
        loss_fea = 0.0

        w = [1.0, 10.0, 0.0, 1.0]
        total_loss = w[0] * loss_pix + w[1] * loss_gra + w[2] * loss_mean + w[3] * loss_fea
        # -----------------------------------
        outputs = {'out': out,
                   'weight': weight,
                   'middle_temp': middle_temp,
                   'pix_loss': w[0] * loss_pix,
                   'gra_loss': w[1] * loss_gra,
                   'mean_loss': w[2] * loss_mean,
                   'sh_loss': loss_sh,
                   'mi_loss': loss_mi,
                   'de_loss': loss_de,
                   'fea_loss': w[3] * loss_fea,
                   'total_loss': total_loss}

        return outputs


