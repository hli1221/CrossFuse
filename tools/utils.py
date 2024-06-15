# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @Project : vision_transformer_pytorch
# @File : tools.py
# @Time : 2021/5/12 20:22

import random
import torch
import torch.nn as nn
import numpy as np
import cv2
from os import listdir
from os.path import join
import seaborn as sns
import logging
import math
import matplotlib.pyplot as plt
from args_trans import Args as args

EPSILON = 1e-6


def list_images_datasets(directory, num):
    images = []
    names = []
    n = len(directory)
    for i in range(n):
        index = 0
        dir_temp = directory[i]
        dir = listdir(dir_temp)
        dir.sort()
        for file in dir:
            # name = file.lower()
            if i == 0 and index > num:
                break
            name = file
            if name.endswith('.png'):
                images.append(join(dir_temp, file))
            elif name.endswith('.jpg'):
                images.append(join(dir_temp, file))
            elif name.endswith('.jpeg'):
                images.append(join(dir_temp, file))
            elif name.endswith('.bmp'):
                images.append(join(dir_temp, file))
            elif name.endswith('.tif'):
                images.append(join(dir_temp, file))
            # name1 = name.split('.')
            index += 1
            names.append(name)
        
    return images, names


def list_images_test(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        # name = file.lower()
        name = file
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        # name1 = name.split('.')
        names.append(name)
    
    return images, names


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 4 epochs"""
    # lr *= (0.1 ** (epoch // 2))
    if epoch-1 > 0:
        lr *= 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def getBinaryTensor(tensor, boundary):
    one = torch.ones_like(tensor)
    zero = torch.zeros_like(tensor)
    return torch.where(tensor > boundary, one, zero)


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        # x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


def recons_midle_feature(x_in, start=0):
    b, c, h, w = x_in.shape
    # x = x[:, 0:64, :, :]
    # x = x.view(b, 1, h*8, w*8)
    h_patches = w_patches = 8
    x = torch.abs(x_in)
    # x = x_in
    x = normalize_tensor(x)
    x = x * 255

    patch_matrix = None
    for i in range(h_patches):
        raw_img = None
        for j in range(h_patches):
            patch_one = x[:, i * w_patches + j + start, :, :]
            patch_one = patch_one.view(b, 1, h, w)
            # patch_one = torch.abs(patch_one)
            # patch_one = normalize_tensor(patch_one) * 255
            # patch_one = np.reshape(patch_one, [1, c, c_h, c_w])
            if j == 0:
                raw_img = patch_one
            else:
                raw_img = torch.cat((raw_img, patch_one), 2)
        if i == 0:
            patch_matrix = raw_img
        else:
            patch_matrix = torch.cat((patch_matrix, raw_img), 3)
    # for i in range(6,7):
    #     raw_img = None
    #     for j in range(2,3):
    #         patch_one = x[:, i * w_patches + j + start, :, :]
    #         patch_one = patch_one.view(b, 1, h, w)
    #         # raw_img = patch_one
    #         patch_matrix = patch_one

    return patch_matrix


def recons_midle_feature_two(x_in, y_in, start=0):
    b, c, h, w = x_in.shape
    # x = x[:, 0:64, :, :]
    # x = x.view(b, 1, h*8, w*8)
    h_patches = w_patches = 8
    # x = (normalize_tensor(torch.abs(x_in))) * 255
    # y = (normalize_tensor(torch.abs(y_in))) * 255
    x = x_in
    y = y_in

    patch_matrix_x = None
    patch_matrix_y = None
    for i in range(3,4):
        raw_img = None
        for j in range(3,4):
            patch_one = x[:, i * w_patches + j + start, :, :]
            patch_one = patch_one.view(b, 1, h, w)
            # raw_img = patch_one
            patch_matrix_x = patch_one
            
    for i in range(6,7):
        raw_img = None
        for j in range(2,3):
            patch_ = y[:, i * w_patches + j + start, :, :]
            patch_ = patch_.view(b, 1, h, w)
            # raw_img = patch_one
            patch_matrix_y = patch_
    z = patch_matrix_x + patch_matrix_y
    z = (normalize_tensor(torch.abs(z))) * 255

    return z


def save_image_heat_map_two(x, y, output_path):
    
    img_fusion = recons_midle_feature_two(x, y)
    
    # img_fusion = normalize_tensor(torch.mean(torch.abs(img_fusion), dim=1, keepdim=True)) * 255
    # img_fusion = normalize_tensor(torch.mean(img_fusion, dim=1, keepdim=True)) * 255
    img_fusion = img_fusion.cpu().data[0].numpy()

    if len(img_fusion.shape) > 2:
        img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = img_fusion.astype('uint8')
    
    img_fusion=cv2.applyColorMap(img_fusion, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, img_fusion)
    # return img_fusion


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def img_padding(x, c_h, c_w):
    c, h, w = x.shape
    h_patches = int(np.ceil(h / c_h))
    w_patches = int(np.ceil(w / c_w))

    h_padding = h_patches * c_h - h
    w_padding = w_patches * c_w - w
    # reflect, symmetric, wrap, edge, linear_ramp, maximum, mean, median, minimum
    x = np.pad(x, ((0, 0), (0, h_padding), (0, w_padding)), 'reflect')
    return x, [h_patches, w_patches, h_padding, w_padding]


def crop_op(img, c_h, c_w):
    img, pad_para = img_padding(img, c_h, c_w)
    c, h, w = img.shape
    h_patches = pad_para[0]
    w_patches = pad_para[1]
    # -------------------------------------------
    patch_matrix = None
    for i in range(h_patches):
        for j in range(w_patches):
            patch_one = img[:, i * c_h: (i + 1) * c_h, j * c_w: (j + 1) * c_w]
            patch_one = np.reshape(patch_one, [1, c, c_h, c_w])
            if i == 0 and j == 0:
                patch_matrix = patch_one
            else:
                patch_matrix = np.concatenate((patch_matrix, patch_one), 0)

    return patch_matrix, pad_para


# load images
def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        mode = cv2.IMREAD_COLOR
    else:
        mode = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(path, mode)
    # -----------------------------------------------------
    assert image is not None, \
        f"The type of image ({path}) is None."
    # -----------------------------------------------------
    if height is not None and width is not None:
        image = cv2.resize(image, (height, width))
    return image


# get training images
def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


#  --------------------------------------------------------------------
def get_test_images_color(paths, height=256, width=256, flag=True):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    images_cb = []
    images_cr = []
    for path in paths:
        image = get_image(path, height, width, flag)
        image_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        image_y = image_ycbcr[:, :, 0]
        image_cb = image_ycbcr[:, :, 1]
        image_cr = image_ycbcr[:, :, 2]

        image_y = np.reshape(image_y, [1, image_y.shape[0], image_y.shape[1]])
        image_cb = np.reshape(image_cb, [image.shape[0], image.shape[1], 1])
        image_cr = np.reshape(image_cr, [image.shape[0], image.shape[1], 1])
        
        images.append(image_y)
        images_cb.append(image_cb)
        images_cr.append(image_cr)
        
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images, images_cb, images_cr


# get testing images
def get_test_images(paths, crop_h=256, crop_w=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    pad_para = None
    for path in paths:
        image = get_image(path, None, None, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        image_crop, pad_para = crop_op(image, crop_h, crop_w)

        images.append(image_crop)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images, pad_para


def save_image(img_fusion, output_path):
    img_fusion = img_fusion.cpu().data[0].numpy()

    if len(img_fusion.shape) > 2:
        img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = img_fusion.astype('uint8')
    cv2.imwrite(output_path, img_fusion)
    

# def save_image_heat(img_fusion, output_path):
#     img_fusion = img_fusion.cpu().data[0].numpy()

#     if len(img_fusion.shape) > 2:
#         img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
#     else:
#         img_fusion = img_fusion.astype('uint8')
#     heat_img = cv2.applyColorMap(img_fusion, cv2.COLORMAP_JET)
#     cv2.imwrite(output_path, heat_img)


def save_image_heat_map_list(fea_list, output_path):
    fea_all = []
    for i, fea in enumerate(fea_list):
        fea_array = save_image_heat_map(fea, output_path)
        if i == 0:
            fea_all = fea_array
        else:
            fea_all = np.concatenate((fea_all, fea_array), 1)
    # fea_all = np.abs(fea_all)
    # fea_all = (fea_all - np.min(fea_all)) / (np.max(fea_all) - np.min(fea_all) + EPSILON)
    # fea_all = fea_all * 255
    fea_all = fea_all.astype('uint8')
    
    fea_all=cv2.applyColorMap(fea_all, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, fea_all)
    

def save_image_heat_map(img_fusion, output_path):
    
    # img_fusion = recons_midle_feature(img_fusion)
    
    img_fusion = normalize_tensor(torch.mean(torch.abs(img_fusion), dim=1, keepdim=True)) * 255
    # img_fusion = normalize_tensor(torch.mean(img_fusion, dim=1, keepdim=True)) * 255
    img_fusion = img_fusion.cpu().data[0].numpy()

    if len(img_fusion.shape) > 2:
        img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = img_fusion.astype('uint8')
    
    img_fusion=cv2.applyColorMap(img_fusion, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, img_fusion)
    # return img_fusion
    

def save_image_color(img_fusion, vi_cb, vi_cr, output_path):
    img_fusion = img_fusion.cpu().data[0].numpy()

    if len(img_fusion.shape) > 2:
        img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = img_fusion.astype('uint8')
    img_fusion = np.reshape(img_fusion, [img_fusion.shape[0], img_fusion.shape[1], 1])
    img = [img_fusion, vi_cb[0], vi_cr[0]]
    img = np.squeeze(np.stack(img, axis=2))
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite(output_path, img)


def save_image_with_pad(outputs, output_path, pad_para):
    h_patches = pad_para[0]
    w_patches = pad_para[1]
    h_paddings = pad_para[2]
    w_paddings = pad_para[3]
    assert len(outputs) == h_patches * w_patches , \
        f"The number of output patches ({len(outputs)}) doesn't match the crop operation ({h_patches}*{w_patches})."
    final_img = None
    for i in range(h_patches):
        raw_img = None
        for j in range(w_patches):
            patch = outputs[i * w_patches + j]
            patch = patch.cpu().data[0].numpy()
            if j == 0:
                raw_img = patch
            else:
                raw_img = np.concatenate((raw_img, patch), 2)
        if i == 0:
            final_img = raw_img
        else:
            final_img = np.concatenate((final_img, raw_img), 1)

    if h_paddings == 0 and w_paddings != 0:
        final_img = final_img[:, :, :-w_paddings]
    elif h_paddings != 0 and w_paddings == 0:
        final_img = final_img[:, :-h_paddings, :]
    elif h_paddings == 0 and w_paddings == 0:
        final_img = final_img[:, :, :]
    else:
        final_img = final_img[:, :-h_paddings, :-w_paddings]

    if len(final_img.shape) > 2:
        img_fusion = final_img.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = final_img.astype('uint8')
    cv2.imwrite(output_path, img_fusion)


def normalize_tensor(tensor):
    (b, ch, h, w) = tensor.size()
    # tensor_v = tensor.contiguous().view(b, -1)
    tensor_v = tensor.reshape(b, ch, h * w)
    t_min = torch.min(tensor_v, 2)[0]
    t_max = torch.max(tensor_v, 2)[0]

    t_min = t_min.view(b, ch, 1, 1)
    t_min = t_min.repeat(1, 1, h, w)
    t_max = t_max.view(b, ch, 1, 1)
    t_max = t_max.repeat(1, 1, h, w)
    tensor = (tensor - t_min) / (t_max - t_min + EPSILON)
    return tensor


def vision_features(features, img_type, fea_type):
    file_name = 'feature_maps_' + img_type + '_' + fea_type + '.png'
    output_path = './output/feature_maps/' + file_name

    h = w = int(np.sqrt(features.size(1)))
    map_all = None
    for idx_h in range(h):
        map_raw = None
        for idx_w in range(w):
            index = idx_h * w + idx_w
            map = features[0, index, :, :].view(1, 1, features.size(2), features.size(3))
            map = normalize_tensor(map)
            if idx_w == 0:
                map_raw = map
            else:
                map_raw = torch.cat((map_raw, map), 3)
        if idx_h == 0:
            map_all = map_raw
        else:
            map_all = torch.cat((map_all, map_raw), 2)

    # map_all = map_all * 255
    # save images
    # save_image(map_all, output_path)
    show_heatmap(map_all, output_path)


def show_heatmap(feature, output_path):
    sns.set()
    feature = feature.float()
    if args.cuda:
        feature = feature.cpu().data[0].numpy()
    else:
        feature = feature.clamp(0, 255).data[0].numpy()

    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + EPSILON)
    feature = feature * 255
    feature = feature.transpose(1, 2, 0).astype('uint8')
    if feature.shape[2] == 1:
        feature = feature.reshape([feature.shape[0], feature.shape[1]])

    img_fusion=cv2.applyColorMap(feature, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, img_fusion)
    
    # fig = plt.figure()
    # # sns.heatmap(feature, cmap='YlGnBu', xticklabels=50, yticklabels=50)
    # sns.heatmap(feature, xticklabels=50, yticklabels=50)
    # fig.savefig(output_path, bbox_inches='tight')
    

