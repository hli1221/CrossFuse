# -*- encoding: utf-8 -*-
'''
@Author  :   Hui Li, Jiangnan University
@Contact :   lihui.cv@jiangnan.edu.cn
@File    :   args_auto.py
@Time    :   2024/06/15 16:29:31
'''

# here put the import lib

class Args():
	# For training
	path = ['G:/datasets/Image-fusion/KAIST/lwir/'] # for type_flag = 'ir'
	# path = ['/data/Disk_B/KAIST-RGBIR/visible/'] # for type_flag = 'vi'
	type_flag = 'ir' # or 'vi'
	cuda = True
	lr = 0.0001
	epochs = 4
	batch = 2
	step = 10
	w = [1.0, 10000.0, 0.1, 1.0]
	train_num = 40000
	# Network Parameters
	channel = 1
	Height = 256
	Width = 256
	crop_h = 256
	crop_w = 256

	resume_model_auto = None
	save_auto_model = "./models/autoencoder"







