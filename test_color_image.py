# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project: CrossFuse
# @File: test_color_image
# @Time: 2023/3/9 15:27


import os
import torch
import numpy as np
from torch.autograd import Variable
from network.net_autoencoder import Auto_Encoder_single
from network.net_conv_trans import Trans_FuseNet
from tools import utils
from args_trans import Args as args

k = 10


# imagenet_labels = dict(enumerate(open("classes.txt")))


def load_model(custom_config_auto, custom_config_trans, model_path_auto_ir, model_path_auto_vi, model_path_trans):
    model_auto_ir = Auto_Encoder_single(**custom_config_auto)
    model_auto_ir.load_state_dict(torch.load(model_path_auto_ir))
    model_auto_ir.cuda()
    model_auto_ir.eval()
    
    model_auto_vi = Auto_Encoder_single(**custom_config_auto)
    model_auto_vi.load_state_dict(torch.load(model_path_auto_vi))
    model_auto_vi.cuda()
    model_auto_vi.eval()
    # ---------------------------------------------------------
    model_trans = Trans_FuseNet(**custom_config_trans)
    model_trans.load_state_dict(torch.load(model_path_trans))
    model_trans.cuda()
    model_trans.eval()
    return model_auto_ir, model_auto_vi, model_trans


def test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path,
         output_path_fea):

    ir_img = utils.get_train_images(ir_path, None, None, flag=False)
    vi_img, vi_cb, vi_cr = utils.get_test_images_color(vi_path, None, None, flag=img_flag)
    ir_img = Variable(ir_img, requires_grad=False)
    vi_img = Variable(vi_img, requires_grad=False)
    if args.cuda:
        ir_img = ir_img.cuda()
        vi_img = vi_img.cuda()
    
    # ---------------------------------------------
    # outputs = model.reconsturce(ir_img, vi_img)
    ir_sh, ir_de = model_auto_ir(ir_img)
    vi_sh, vi_de = model_auto_vi(vi_img)
    outputs = model_trans(ir_de, ir_sh, vi_de, vi_sh, shift_flag)
    img_out = outputs['out']
    ir_self = outputs['ir_self']
    vi_self = outputs['vi_self']
    fuse_cross = outputs['fuse_cross']
    # # ---------------------------------------------
    
    # ---------------------------------------------
    path_out = output_path + '/results_crossfuse_'
    path_out_fea = output_path_fea + '/result_crossfuse_'
    utils.save_image_color(img_out, vi_cb, vi_cr, path_out + ir_name)
    utils.save_image(ir_self, path_out_fea + 'irself_' + ir_name)
    utils.save_image(vi_self, path_out_fea + 'viself_' + ir_name)
    utils.save_image(fuse_cross, path_out_fea + 'cross_' + ir_name)
    
    print('Done. ', ir_name)


if __name__ == "__main__":
    # Auto-Encoder
    custom_config_auto = {
        "in_channels": 1,
        "out_channels": 1,
        "en_out_channels1": 32,
        "en_out_channels": 64,
        "num_layers": 3,
        "dense_out": 128,
        "part_out": 128,
        "train_flag": False,
    }
    # Trans module
    custom_config_trans = {
        "en_out_channels1": 32,
        "out_channels": 1,
        "part_out": 128,
        "train_flag": False,
        
        "img_size": 32,
        "patch_size": 2,
        "depth_self": 1,
        "depth_cross": 1,
        "n_heads": 16,
        "qkv_bias": True,
        "mlp_ratio": 4,
        "p": 0.,
        "attn_p": 0.,
    }
    
    resume_model_auto_ir = "./models/autoencoder/auto_encoder_epoch_5_ir.model"
    resume_model_auto_vi = "./models/autoencoder/auto_encoder_epoch_5_vi.model"
    
    # model_path_auto = "./models/autoencoder/auto_encoder_epoch_3.model"
    # model_path_trans = "./models/transfuse/fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model"
    model_path_trans = "./models/transfuse/fusetrans_epoch_8_nosh.model"
    # ----------------------------------------------------
    img_flag = True

    test_path_ir = './images/M3FD_Fusion/ir'
    test_path_vi = './images/M3FD_Fusion/vis'
    data_type = '/M3FD_Fusion_nosh'
 
    # test_path_ir = './images/vot/ir'
    # test_path_vi = './images/vot/vis'
    # data_type = '/vot_transfuse'
    
    ir_pathes, ir_names = utils.list_images_test(test_path_ir)
    # ---------------------------------------------------
    output_path1 = './output/crossfuse_test'
    if os.path.exists(output_path1) is False:
        os.mkdir(output_path1)
    output_path = output_path1 + data_type
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    output_path_fea = output_path + '/feature'
    if os.path.exists(output_path_fea) is False:
        os.mkdir(output_path_fea)
    # ---------------------------------------------------
    count = 0
    shift_flag = True
    with torch.no_grad():
        model_auto_ir, model_auto_vi, model_trans = load_model(custom_config_auto, custom_config_trans,
                                                               resume_model_auto_ir, resume_model_auto_vi,
                                                               model_path_trans)
        for ir_name in ir_names:
            # vi_name = ir_name.replace('IR', 'VIS')
            vi_name = ir_name
            ir_path = os.path.join(test_path_ir, ir_name)
            vi_path = os.path.join(test_path_vi, vi_name)
            # ---------------------------------------------------
            # if vi_name.__contains__('11'):
            test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path,
                 output_path_fea)




