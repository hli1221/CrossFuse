# -*- encoding: utf-8 -*-
'''
@Author  :   Hui Li, Jiangnan University
@Contact :   lihui.cv@jiangnan.edu.cn
@File    :   transformer_cam.py
@Time    :   2023/03/30 18:00:20
'''

import torch
import torch.nn as nn
import numpy as np
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tools.utils import vision_features, save_image_heat_map, save_image_heat_map_list


class Padding_tensor(nn.Module):
    def __init__(self, patch_size):
        super(Padding_tensor, self).__init__()
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
    
    
class PatchEmbed_tensor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.padding_tensor = Padding_tensor(patch_size)

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
                patch_one = patch_one.reshape(-1, c, 1, self.patch_size, self.patch_size)
                if i == 0 and j == 0:
                    patch_matrix = patch_one
                else:
                    patch_matrix = torch.cat((patch_matrix, patch_one), dim=2)
        # patch_matrix  # (b, c, N, patch_size, patch_size)
        return patch_matrix, patches_paddings
    
    
class Recons_tensor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, patches_tensor, patches_paddings):
        B, C, N, Ph, Pw = patches_tensor.shape
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        h_padding = patches_paddings[2]
        w_padding = patches_paddings[3]
        assert N == h_patches * w_patches, \
            f"The number of patches ({N}) doesn't match the Patched_embed operation ({h_patches}*{w_patches})."
        assert Ph == self.patch_size and Pw == self.patch_size, \
            f"The size of patch tensor ({Ph}*{Pw}) doesn't match the patched size ({self.patch_size}*{self.patch_size})."

        patches_tensor = patches_tensor.view(-1, C, N, self.patch_size, self.patch_size)
        # ----------------------------------------
        pic_all = None
        for i in range(h_patches):
            pic_c = None
            for j in range(w_patches):
                if j == 0:
                    pic_c = patches_tensor[:, :, i * w_patches + j, :, :]
                else:
                    pic_c = torch.cat((pic_c, patches_tensor[:, :, i * w_patches + j, :, :]), dim=3)
            if i == 0:
                pic_all = pic_c
            else:
                pic_all = torch.cat((pic_all, pic_c), dim=2)
        b, c, h, w = pic_all.shape
        pic_all = pic_all[:, :, 0:(h-h_padding), 0:(w-w_padding)]
        return pic_all
# -----------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # x = x + 1e-6
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


# self or cross attention
class Attention(nn.Module):
    def __init__(self, dim, n_heads=16, qkv_bias=True, attn_p=0., proj_p=0., cross=False):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # self.recons_tensor = Recons_tensor(2)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.cross = cross
        if cross:
            self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):

        if self.cross:
            n_samples, n_tokens, dim = x[0].shape
            if dim != self.dim:
                raise ValueError

            n_tokens_en = n_tokens
            q = self.q_linear(x[0]).reshape(n_samples, n_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_linear(x[1]).reshape(n_samples, n_tokens_en, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_linear(x[2]).reshape(n_samples, n_tokens_en, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            n_samples, n_tokens, dim = x.shape
            if dim != self.dim:
                raise ValueError

            qkv = self.qkv(x)  # (n_samples, n_patches, 3 * dim)
            qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
            )  # (n_smaples, n_patches, 3, n_heads, head_dim)
            qkv = qkv.permute(
                2, 0, 3, 1, 4
            )  # (3, n_samples, n_heads, n_patches, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches, n_patches)
        # exp(-x)
        # dp 过大，softmax之后数值可能溢出
        if self.cross:
            # t_str = time.time()
            # dp_s = dp.softmax(dim=-1)
            # vision_features(dp_s, 'atten', 'dp_'+str(t_str))
            dp = -1 * dp
            # attn = dp.softmax(dim=-1)
            # vision_features(attn, 'atten', 'dp_v_'+str(t_str))
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches, n_patches)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        # if self.cross:
        #     x_temp = x.view(1, 256, 128, 2, 2).permute(0, 2, 1, 3, 4)
        #     x_temp = self.recons_tensor(x_temp, [16,16,0,0])  # B, C, H, W
        #     vision_features(x_temp, 'atten', 'attn_x')
        
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0., cross=False):
        super().__init__()
        self.cross = cross
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p,
            cross=cross
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        if self.cross:
            x_ = [self.norm1(_x) for _x in x]
            # x_ = x
            out = x[2] + self.attn(x_)
            out = out + self.mlp(self.norm2(out))
            out = [x_[0], out, out]
        else:
            out = x + self.attn(self.norm1(x))
            out = out + self.mlp(self.norm2(out))
        
        return out
# --------------------------------------------------------------------------------------


class self_atten_module(nn.Module):
    def __init__(self, embed_dim, num_p, depth, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, n_heads=n_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p, cross=False)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x_in):
        # x_ori = x_in
        x = x_in
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x_self = x
        # x_self = x_in + x
        return x_self


class cross_atten_module(nn.Module):
    def __init__(self, embed_dim, num_patches, depth, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                      cross=True)
                if i == 0 else
                Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                      cross=True)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x1_ori, x2_ori):
        x1 = x1_ori
        x2 = x2_ori
        x2 = self.pos_drop(x2)
        x = [x1, x2, x2]
        for block in self.blocks:
            x = block(x)
            x[2] = self.norm(x[2])
        x_self = x[2]
        # x_self = x2_ori + x[2]
        return x_self
    

class self_atten(nn.Module):
    def __init__(self, patch_size, embed_dim, num_patches, depth_self, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_embed_tensor = PatchEmbed_tensor(patch_size)
        self.recons_tensor = Recons_tensor(patch_size)
        self.self_atten1 = self_atten_module(embed_dim, num_patches, depth_self,
                                              n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten2 = self_atten_module(embed_dim, num_patches, depth_self,
                                                   n_heads, mlp_ratio, qkv_bias, p, attn_p)

    def forward(self, x1, x2, last=False):
        # patch
        x_patched1, patches_paddings = self.patch_embed_tensor(x1)
        # B, C, N, Ph, Pw = x_patched1.shape
        x_patched2, _ = self.patch_embed_tensor(x2)
        # B, C, N, Ph, Pw = x_patched1.shape
        b, c, n, h, w = x_patched1.shape
        # b, n, c*h*w
        x_patched1 = x_patched1.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x_patched2 = x_patched2.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x1_self_patch = self.self_atten1(x_patched1)
        x2_self_patch = self.self_atten2(x_patched2)
       
        # reconstruct
        if last is False:
            x1_self_patch = x1_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
            x_self1 = self.recons_tensor(x1_self_patch, patches_paddings)  # B, C, H, W
            x2_self_patch = x2_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
            x_self2 = self.recons_tensor(x2_self_patch, patches_paddings)  # B, C, H, W
        else:
            x_self1 = x1_self_patch
            x_self2 = x2_self_patch

        return x_self1, x_self2, patches_paddings


class cross_atten(nn.Module):
    def __init__(self, patch_size, embed_dim, num_patches, depth_self, depth_cross, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_embed_tensor = PatchEmbed_tensor(patch_size)
        self.recons_tensor = Recons_tensor(patch_size)
        
        self.cross_atten1 = cross_atten_module(embed_dim, num_patches, depth_cross,
                                                     n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.cross_atten2 = cross_atten_module(embed_dim, num_patches, depth_cross,
                                                     n_heads, mlp_ratio, qkv_bias, p, attn_p)
        # self.cross_atten = patch_cross_atten_module(img_size, patch_size, embed_dim, num_patches, depth_cross,
        #                                              n_heads, mlp_ratio, qkv_bias, p, attn_p)

    def forward(self, x1, x2, patches_paddings):
        # patch
        x_patched1, patches_paddings = self.patch_embed_tensor(x1)
        # B, C, N, Ph, Pw = x_patched1.shape
        x_patched2, _ = self.patch_embed_tensor(x2)
        # B, C, N, Ph, Pw = x_patched1.shape
        b, c, n, h, w = x_patched1.shape
        # b, n, c*h*w
        x1_self_patch = x_patched1.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x2_self_patch = x_patched2.transpose(2, 1).contiguous().view(b, n, c * h * w)
        
        x_in1 = x1_self_patch
        x_in2 = x2_self_patch
        cross1 = self.cross_atten1(x_in1, x_in2)
        cross2 = self.cross_atten2(x_in2, x_in1)
        out = cross1 + cross2
        
        # reconstruct
        x1_self_patch = x1_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        x_self1 = self.recons_tensor(x1_self_patch, patches_paddings)  # B, C, H, W
        x2_self_patch = x2_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        x_self2 = self.recons_tensor(x2_self_patch, patches_paddings)  # B, C, H, W
        
        cross1 = cross1.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        cross1_all = self.recons_tensor(cross1, patches_paddings)  # B, C, H, W
        
        cross2 = cross2.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        cross2_all = self.recons_tensor(cross2, patches_paddings)  # B, C, H, W
        
        out = out.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        out_all = self.recons_tensor(out, patches_paddings)  # B, C, H, W
        
        return out_all, x_self1, x_self2, cross1_all, cross2_all


class cross_encoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_patches, depth_self, depth_cross, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.shift_size = int(img_size / 2)
        self.depth_cross = depth_cross
        # self.depth_cross = 0

        self.self_atten_block1 = self_atten(self.patch_size, embed_dim, num_patches, depth_self,
                                              n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten_block2 = self_atten(self.patch_size, embed_dim, num_patches, depth_self,
                                                   n_heads, mlp_ratio, qkv_bias, p, attn_p)
        
        self.cross_atten_block = cross_atten(self.patch_size, embed_dim, self.num_patches, depth_self,
                                               depth_cross, n_heads, mlp_ratio, qkv_bias, p, attn_p)

    def forward(self, x1, x2, shift_flag=True):
        # x1 -->> ir, x2 -->> vi
        # self-attention
        x1_atten, x2_atten, paddings = self.self_atten_block1(x1, x2)
        x1_a, x2_a = x1_atten, x2_atten
        # shift
        if shift_flag:
            shifted_x1 = torch.roll(x1_atten, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shifted_x2 = torch.roll(x2_atten, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            x1_atten, x2_atten, _ = self.self_atten_block2(shifted_x1, shifted_x2)
            roll_x_self1 = torch.roll(x1_atten, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            roll_x_self2 = torch.roll(x2_atten, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x1_atten, x2_atten, _ = self.self_atten_block2(x1_atten, x2_atten)
            roll_x_self1 = x1_atten
            roll_x_self2 = x2_atten
        # -------------------------------------
        # cross attention
        if self.depth_cross > 0:
            out, x_self1, x_self2, x_cross1, x_cross2 = self.cross_atten_block(roll_x_self1, roll_x_self2, paddings)
        else:
            out = roll_x_self1 + roll_x_self2
            x_self1, x_self2, x_cross1, x_cross2 = roll_x_self1, roll_x_self2, roll_x_self1, roll_x_self2
        # -------------------------------------
        # recons
        return out, x1_a, x2_a, roll_x_self1, roll_x_self2, x_cross1, x_cross2
