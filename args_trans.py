# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse
# @File : args_trans.py
# @Time : 2021/11/9 14:15

class Args():
	# For training
	path_ir = ['G:/datasets/Image-fusion/KAIST/lwir/']
	# path_ir = ['/data/Disk_B/KAIST-RGBIR/lwir/']
	cuda = True
	lr = 0.001
	epochs = 32
	batch = 8
	train_num = 20000
	step = 10
	# Network Parameters
	channel = 1
	Height = 256
	Width = 256
 
	crop_h = 256
	crop_w = 256

	vgg_model_dir = "./models/vgg"
	resume_model_auto_ir = "./models/autoencoder/auto_encoder_epoch_5_ir.model"
	resume_model_auto_vi = "./models/autoencoder/auto_encoder_epoch_5_vi.model"
	# resume_model_auto_ir = None
	# resume_model_auto_vi = None

	# resume_model_trans = "./models/transfuse/fusetrans_epoch_4.model"
	resume_model_trans = None
	save_fusion_model = "./models"
	save_loss_dir = "./models/loss"