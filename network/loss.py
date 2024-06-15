# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse
# @File : loss.py
# @Time : 2021/11/8 18:36

import torch
import torch.nn as nn
import numpy as np
import pytorch_msssim
import tools.utils as utils
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
ssim_loss = pytorch_msssim.msssim

EPSILON = 1e-6


class Gradient_loss(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # average
        patch_szie = 3
        reflection_padding = int(np.floor(patch_szie / 2))
        weight = torch.ones([1, 1, patch_szie, patch_szie])
        self.conv_avg = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
                                  stride=1, padding=reflection_padding, bias=False)
        self.conv_avg.weight.data = (1 / (patch_szie * patch_szie)) * weight.repeat(channels, channels, 1, 1).float()
        self.conv_avg.requires_grad_(False)
        
        # self.conv_one = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        weight = torch.from_numpy(np.array([[[[0.,  1., 0.], 
                                              [1., -4., 1.], 
                                              [0.,  1., 0.]]]]))
        self.conv_two = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_two.weight.data = weight.repeat(channels, channels, 1, 1).float()
        self.conv_two.requires_grad_(False)

        # LoG
        weight_log = torch.from_numpy(np.array([[[[0., 0., -1, 0., 0.],
                                                  [0., -1, -2, -1, 0.],
                                                  [-1, -2, 16, -2, -1],
                                                  [0., -1, -2, -1, 0.],
                                                  [0., 0., -1, 0., 0.]]]]))
        # weight_log = torch.from_numpy(np.array([[[[0., 0., 0., -1, 0., 0., 0.],
        #                                           [0., 0., -1, -2, -1, 0., 0.],
        #                                           [0., -1, -2, -3, -2, -1, 0.],
        #                                           [-1, -2, -3, 40, -3, -2, -1],
        #                                           [0., -1, -2, -3, -2, -1, 0.],
        #                                           [0., 0., -1, -2, -1, 0., 0.],
        #                                           [0., 0., 0., -1, 0., 0., 0.]]]]))
        self.conv_log = torch.nn.Conv2d(channels, channels, (5, 5), stride=1, padding=3, bias=False)
        self.conv_log.weight.data = weight_log.repeat(channels, channels, 1, 1).float()
        self.conv_log.requires_grad_(False)

        # sobel
        weight_s1 = torch.from_numpy(np.array([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]))
        weight_s2 = torch.from_numpy(np.array([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]]]))
        self.conv_sx = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_sx.weight.data = weight_s2.repeat(channels, channels, 1, 1).float()
        self.conv_sx.requires_grad_(False)
        self.conv_sy = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_sy.weight.data = weight_s1.repeat(channels, channels, 1, 1).float()
        self.conv_sy.requires_grad_(False)

        # average
        patch_szie = 3
        reflection_padding = int(np.floor(patch_szie / 2))
        weight_avg = torch.ones([1, 1, patch_szie, patch_szie])
        self.conv_avg = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
                                  stride=1, padding=reflection_padding, bias=False)
        self.conv_avg.weight.data = (1 / (patch_szie * patch_szie)) * weight_avg.repeat(channels, channels, 1, 1).float()
        self.conv_avg.requires_grad_(False)

    def forward(self, out, x_ir, x_vi):
        channels = x_ir.size()[1]
        channels_t = out.size()[1]
        assert channels == channels_t, \
            f"The channels of x ({channels}) doesn't match the channels of target ({channels_t})."
        g_o = torch.clamp(self.conv_two(out), min = 0)
        g_xir = torch.clamp(self.conv_two(x_ir), min = 0)
        g_xvi = torch.clamp(self.conv_two(x_vi), min = 0)
        g_sub = g_o
        
        # LoG
        # g_o = torch.abs(self.conv_log(out))
        # g_xir = torch.abs(self.conv_log(x_ir))
        # g_xvi = torch.abs(self.conv_log(x_vi))
        # g_sub = g_o
        # loss = mse_loss(g_sub, g_xvi)
        # sobel
        # g_t1 = self.conv_sx(target)
        # g_x1 = self.conv_sx(x)
        # g_t2 = self.conv_sy(target)
        # g_x2 = self.conv_sy(x)
        # loss = mse_loss(g_t1, g_x1) + mse_loss(g_t2, g_x2)

        # g_o = torch.abs(out - self.conv_avg(out))
        # g_xir = torch.abs(x_ir - self.conv_avg(x_ir))
        # g_xvi = torch.abs(x_vi - self.conv_avg(x_vi))
        # # g_sub = torch.abs(g_o - g_xir)
        # g_sub = g_o
        # loss = mse_loss(g_sub, g_xvi)
        
        # t_one = torch.ones_like(g_xir)
        # mask = torch.clamp(g_xir - g_xvi, min=0.0)
        # mask = torch.where(mask > 0, t_one, mask)
        # g_target = mask * g_xir + (1 - mask) * g_xvi
        g_target = torch.max(g_xir, g_xvi)
        loss = mse_loss(g_sub, g_target)
        # loss = l1_loss(g_sub, g_target)
        
        return loss, g_sub, g_xir, g_xvi, g_target


class Order_loss(nn.Module):
    def __init__(self, channels, patch_szie=11):
        super().__init__()

        reflection_padding = int(np.floor(patch_szie / 2))
        weight = torch.ones([1, 1, patch_szie, patch_szie])
        self.conv_two = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
                                  stride=1, padding=reflection_padding, bias=False)
        self.conv_two.weight.data = (1 / (patch_szie * patch_szie)) * weight.repeat(channels, channels, 1, 1).float()
        self.conv_two.requires_grad_(False)

        # LoG
        weight_log = torch.from_numpy(np.array([[[[0., 0., -1, 0., 0.],
                                                  [0., -1, -2, -1, 0.],
                                                  [-1, -2, 16, -2, -1],
                                                  [0., -1, -2, -1, 0.],
                                                  [0., 0., -1, 0., 0.]]]]))
        self.conv_log = torch.nn.Conv2d(channels, channels, (5, 5), stride=1, padding=2, bias=False)
        self.conv_log.weight.data = weight_log.repeat(channels, channels, 1, 1).float()
        self.conv_log.requires_grad_(False)

    def forward(self, out, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        # g_x = torch.sqrt((x - self.conv_two(x)) ** 2)
        # g_y = torch.sqrt((y - self.conv_two(y)) ** 2)
        # LoG
        # i_x = self.conv_two(x)
        # i_x = (i_x - torch.min(i_x)) / (torch.max(i_x) - torch.min(i_x))
        # g_x = torch.clamp(self.conv_log(x), min = 0)
        # g_x = (g_x - torch.min(g_x)) / (torch.max(g_x) - torch.min(g_x))
        # s_x = 0.4 * i_x + 0.6 * g_x
        
        # i_y = self.conv_two(y)
        # i_y = (i_y - torch.min(i_y)) / (torch.max(i_y) - torch.min(i_y))
        # g_y = torch.clamp(self.conv_log(y), min = 0)
        # g_y = (g_y - torch.min(g_y)) / (torch.max(g_y) - torch.min(g_y))
        # s_y = 0.6 * i_y + 0.4 * g_y
        # Avg
        s_x = self.conv_two(x)
        # s_x = (s_x - torch.min(s_x)) / (torch.max(s_x) - torch.min(s_x))
        s_y = self.conv_two(y)
        # s_y = (s_y - torch.min(s_y)) / (torch.max(s_y) - torch.min(s_y))
        w_x = s_x / (s_x + s_y + EPSILON)
        w_y = s_y / (s_x + s_y + EPSILON)
        
        # target = torch.max((w_x + 1) * x, (w_y + 1) * y)
        t_one = torch.ones_like(w_x)
        mask = torch.clamp(w_x - w_y, min=0.0)
        mask = torch.where(mask > 0, t_one, mask)
        target = mask * x + (1 - mask) * y
        loss_p = mse_loss(out, target)
        
        return loss_p, target


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.contiguous().view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def feature_loss(vgg, ir, vi, f):
    f_fea = vgg(f)
    ir_fea = vgg(ir)
    vi_fea = vgg(vi)

    loss_rgb = 0.
    loss_fea = 0.
    loss_gram = 0.
    # feature loss
    t_idx = 0
    w_fea = [0.01, 0.01, 200.0]
    w_ir = [0.0, 2.0, 4.0]
    w_vi = [1.0, 1.0, 1.0]
    for _vi, _ir, _f, w1, w2, w3 in zip(vi_fea, ir_fea, f_fea, w_fea, w_ir, w_vi):
        # (bt, cht, ht, wt) = rgb.size()
        if t_idx == 0:
            loss_rgb += w1 * mse_loss(_f, w2 * _ir + w3 * _vi)
        if t_idx == 1:
            loss_fea += w1 * mse_loss(_f, w2 * _ir + w3 * _vi)
        if t_idx == 2:
            gram_out = gram_matrix(_f)
            gram_ir = gram_matrix(_ir)
            gram_vi = gram_matrix(_vi)
            loss_gram += w1 * mse_loss(gram_out, w2 * gram_ir + w3 * gram_vi)
            # loss_gram += w1 * mse_loss(f, w2 * ir + w3 * rgb)
        t_idx += 1
    return loss_rgb, loss_fea, loss_gram


def nucnorm_loss(x):
    B, C, H, W = x.shape
    x_temp = x + EPSILON
    x_temp = x_temp.view(B * C, H, W)
    loss_nuc = 0.0

    for i in range(B * C):
        loss_nuc += torch.sum(torch.svd(x_temp[i, :, :], compute_uv=True)[1])
    loss_nuc = loss_nuc / (B * C)
    return loss_nuc


class padding_tensor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        b, c, h, w = x.shape
        h_patches = int(np.ceil(h / self.patch_size))
        w_patches = int(np.ceil(w / self.patch_size))

        h_padding = np.abs(h - h_patches * self.patch_size)
        w_padding = np.abs(w - w_patches * self.patch_size)

        reflection_padding = [0, w_padding, 0, h_padding]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x, [h_patches, w_patches, h_padding, w_padding]


class get_patch_tensor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.padding_tensor = padding_tensor(patch_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x, patches_paddings = self.padding_tensor(x)
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        # -------------------------------------------
        patch_matrix = None
        for i in range(h_patches):
            for j in range(w_patches):
                patch_one = x[:, :, i * self.patch_size: (i + 1) * self.patch_size,
                            j * self.patch_size: (j + 1) * self.patch_size]
                # patch_one = patch_one.flatten(1)
                # patch_one = patch_one.unsqueeze(2)
                patch_one = patch_one.reshape(-1, c, 1, self.patch_size * self.patch_size)
                if i == 0 and j == 0:
                    patch_matrix = patch_one
                else:
                    patch_matrix = torch.cat((patch_matrix, patch_one), dim=2)
        # patch_matrix  # (b, c, N, patch_size * patch_size)
        return patch_matrix, patches_paddings


class Patch_loss(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        # patch_size = 16 for image size 256*256
        self.patch_size = patch_size
        self.get_patch = get_patch_tensor(patch_size)

    def forward(self, out, x1, x2):
        # 1 - ir, 2 - vi
        B, C, H, W = out.shape
        # patch_matrix  # (b, c, N, patch_size * patch_size)
        patch0, hw_p0 = self.get_patch(out)
        patch1, hw_p1 = self.get_patch(x1)
        patch2, hw_p2 = self.get_patch(x2)

        b0, c0, n0, p0 = patch0.shape
        b1, c1, n1, p1 = patch1.shape
        b2, c2, n2, p2 = patch2.shape
        assert n0 == n1 == n2 and p0 == p1 == p2, \
                f"The number of patches ({n0}, {n1} and {n2}) or the patch sice ({p0}, {p1} and {p2}) doesn't match ."

        mu1 = torch.mean(patch1, dim=3)
        mu2 = torch.mean(patch2, dim=3)

        mu1_re = mu1.view(b1, c1, n1, 1).repeat(1, 1, 1, p1)
        mu2_re = mu1.view(b2, c2, n2, 1).repeat(1, 1, 1, p2)

        # SD, b1 * c1 * n1 * 1
        sd1 = torch.sqrt(torch.sum(((patch1 - mu1_re) ** 2), dim=3) / p1)
        sd2 = torch.sqrt(torch.sum(((patch2 - mu2_re) ** 2), dim=3) / p2)
        # sd_mask = getBinaryTensor(sd1 - sd2, 0)

        w1 = sd1 / (sd1 + sd2 + EPSILON)
        w2 = sd2 / (sd1 + sd2 + EPSILON)
        w1 = w1.view(b1, c1, n1, 1).repeat(1, 1, 1, p1)
        w2 = w2.view(b2, c2, n2, 1).repeat(1, 1, 1, p2)

        weights = [w1.view(b1, c1, hw_p1[0] * self.patch_size, hw_p1[1] * self.patch_size),
                   w2.view(b1, c1, hw_p1[0] * self.patch_size, hw_p1[1] * self.patch_size)]
        # loss_mu = mse_loss(mu1, mu2)
        # loss_sd = mse_loss(sd1, sd2)
        # out = loss_mu + loss_sd
        out = mse_loss(patch0, w1 * patch1 + w2 * patch2)
        return out, weights



