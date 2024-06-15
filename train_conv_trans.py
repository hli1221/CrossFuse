# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : FuseTrans
# @File : train.py
# @Time : 2021/5/14 14:42

# Train fusion models (CAM and decoder)

import os
import scipy.io as scio
import torch
import time
# pytohn -m visdom.server
# from visdom import Visdom

from tools import utils
import random
from torch.optim import Adam
from torch.autograd import Variable
from network.net_conv_trans import Trans_FuseNet
from network.net_autoencoder import Auto_Encoder_single
from network.loss import Gradient_loss, Order_loss, Patch_loss

from args_trans import Args as args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -------------------------------------------------------
# Auto-Encoder
custom_config_auto = {
	"in_channels": 1,
	"out_channels": 1,
	"en_out_channels1": 32,
	"en_out_channels": 64,
	"num_layers": 3,
	"dense_out": 128,
	"part_out": 128,
	"train_flag": True,
}
# Trans module
custom_config = {
	"en_out_channels1": 32,
	"out_channels": 1,
	"part_out": 128,
	"train_flag": True,
	
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


# -------------------------------------------------------
def load_data(path, train_num):
	# train_num for KAIST
	imgs_path, _ = utils.list_images_datasets(path, train_num)
	# imgs_path = imgs_path[:train_num]
	random.shuffle(imgs_path)
	return imgs_path


# -------------------------------------------------------
def test(model_auto_ir, model_auto_vi, model, shift_flag, e):
	img_flag = False
	test_path_ir = './images/21_pairs_tno/ir'
	test_path_vi = './images/21_pairs_tno/vis'
	ir_pathes, ir_names = utils.list_images_test(test_path_ir)
	# ---------------------------------------------------
	output_path1 = './output/transfuse'
	if os.path.exists(output_path1) is False:
		os.mkdir(output_path1)
	output_path = output_path1 + '/training_21_tno_epoch_' + str(e)
	if os.path.exists(output_path) is False:
		os.mkdir(output_path)
	
	for ir_name in ir_names:
		vi_name = ir_name.replace('IR', 'VIS')
		ir_path = os.path.join(test_path_ir, ir_name)
		vi_path = os.path.join(test_path_vi, vi_name)
		# for training phase
		ir_img = utils.get_train_images(ir_path, height=args.Height, width=args.Width, flag=img_flag)
		vi_img = utils.get_train_images(vi_path, height=args.Height, width=args.Width, flag=img_flag)
		ir_img = Variable(ir_img, requires_grad=False)
		vi_img = Variable(vi_img, requires_grad=False)
		if args.cuda:
			ir_img = ir_img.cuda()
			vi_img = vi_img.cuda()
		
		# ---------------------------------------------
		ir_sh, ir_de = model_auto_ir(ir_img)
		vi_sh, vi_de = model_auto_vi(vi_img)
		outputs = model(ir_de, ir_sh, vi_de, vi_sh, shift_flag)
		out = outputs['out']
		# ---------------------------------------------
		path_out_ir = output_path + '/results_transfuse_' + ir_name
		utils.save_image(out, path_out_ir)


# -------------------------------------------------------
def train(data, img_flag):
	batch_size = args.batch
	step = args.step
	
	model = Trans_FuseNet(**custom_config)
	# model = torch.nn.DataParallel(model_or, list(range(torch.cuda.device_count()))).cuda()
	shift_flag = False
	if args.resume_model_trans is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_model_trans))
		model.load_state_dict(torch.load(args.resume_model_trans))
	# auto-encoder
	model_auto_ir = Auto_Encoder_single(**custom_config_auto)
	model_auto_vi = Auto_Encoder_single(**custom_config_auto)
	# model_auto_ir = torch.nn.DataParallel(model_auto_ir_or, list(range(torch.cuda.device_count()))).cuda()
	# model_auto_vi = torch.nn.DataParallel(model_auto_vi_or, list(range(torch.cuda.device_count()))).cuda()
	
	if args.resume_model_auto_ir is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_model_auto_ir))
		model_auto_ir.load_state_dict(torch.load(args.resume_model_auto_ir))
		model_auto_vi.load_state_dict(torch.load(args.resume_model_auto_vi))
	
	# ------------------------------------------------------
	trainable_params = [{'params': filter(lambda x: x.requires_grad, model.parameters()), 'lr': args.lr}]
	optimizer = Adam(trainable_params, args.lr, weight_decay=0.9)
	# ------------------------------------------------------
	gra_loss = Gradient_loss(custom_config['out_channels'])
	order_loss = Order_loss(custom_config['out_channels'])
	
	# visdom
	# viz = Visdom()
	
	if args.cuda:
		model_auto_ir.cuda()
		model_auto_vi.cuda()
		model.cuda()
		gra_loss.cuda()
		order_loss.cuda()
	
	model_auto_ir.eval()
	model_auto_vi.eval()
	print('Start training.....')
	
	# creating save path
	temp_path_model1 = os.path.join(args.save_fusion_model)
	if os.path.exists(temp_path_model1) is False:
		os.mkdir(temp_path_model1)
	temp_path_model = os.path.join(temp_path_model1, 'transfuse')
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)
	
	temp_path_loss = os.path.join(temp_path_model, 'loss')
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)
	
	loss_p4 = 0.
	loss_p5 = 0.
	loss_p6 = 0.
	loss_p7 = 0.
	loss_p8 = 0.
	loss_p9 = 0.
	loss_p10 = 0.
	loss_all = 0.
	
	loss_mat = []
	model.train()
	count = 0
	for e in range(args.epochs):
		lr_cur = utils.adjust_learning_rate(optimizer, e, args.lr)
		img_paths, batch_num = utils.load_dataset(data, batch_size)
		
		for idx in range(batch_num):
			
			image_paths_ir = img_paths[idx * batch_size:(idx * batch_size + batch_size)]
			img_ir = utils.get_train_images(image_paths_ir, height=args.Height, width=args.Width, flag=img_flag)
			
			image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
			img_vi = utils.get_train_images(image_paths_vi, height=args.Height, width=args.Width, flag=img_flag)
			
			count += 1
			optimizer.zero_grad()
			batch_ir = Variable(img_ir, requires_grad=False)
			batch_vi = Variable(img_vi, requires_grad=False)
			
			if args.cuda:
				batch_ir = batch_ir.cuda()
				batch_vi = batch_vi.cuda()
			
			with torch.no_grad():
				ir_sh, ir_de = model_auto_ir(batch_ir)
				vi_sh, vi_de = model_auto_vi(batch_vi)
			# for DataParallel
			outputs = model.train_module(batch_ir, batch_vi, ir_sh, vi_sh, ir_de, vi_de, shift_flag, gra_loss, order_loss)
			
			img_out = outputs['out']
			weights = outputs['weight']
			# temp = outputs['fuse_temp']
			middle_temp = outputs['middle_temp']
			total_loss = outputs['total_loss']
			loss_mat.append(total_loss.item())
			total_loss.backward()
			optimizer.step()
			
			loss_p4 += outputs['pix_loss']
			loss_p5 += outputs['sh_loss']
			loss_p6 += outputs['mi_loss']
			loss_p7 += outputs['de_loss']
			loss_p8 += outputs['fea_loss']
			loss_p9 += outputs['gra_loss']
			loss_p10 += outputs['mean_loss']
			loss_all += total_loss
			
			# # Test
			# if count % 1000 == 0:
			# 	with torch.no_grad():
			# 		test(model_auto_ir, model_auto_vi, model, e + 1)
			# 		print('Done. Testing image data on epoch {}'.format(e + 1))
			
			if count % step == 0:
				loss_p4 /= step
				loss_p5 /= step
				loss_p6 /= step
				loss_p7 /= step
				loss_p8 /= step
				loss_p9 /= step
				loss_p10 /= step
				loss_all /= step
				# if e == 0 and count == step:
				# 	viz.line([loss_all.item()], [0.], win='train_loss', opts=dict(title='Total Loss'))
				
				mesg = "{} - Epoch {}/{} - Batch {}/{} - lr:{:.6f} - pix loss: {:.6f} - gra loss: {:.6f} - mean loss:{:.6f}" \
				       " - shallow loss: {:.6f} - middle loss: {:.6f}\n" \
				       "deep loss: {:.6f} - fea loss: {:.6f} \t total loss: {:.6f} \n". \
					format(time.ctime(), e + 1, args.epochs, idx + 1, batch_num, lr_cur,
				           loss_p4, loss_p9, loss_p10, loss_p5, loss_p6, loss_p7, loss_p8, loss_all)
				
				# viz.line([loss_all.item()], [count], win='train_loss', update='append')
				img_or1 = torch.cat((batch_ir[0, :, :, :], batch_vi[0, :, :, :]), 0)
				img1 = torch.cat((img_or1, img_out[0, :, :, :]), 0)
				# img_or2 = torch.cat((batch_ir[1, :, :, :], batch_vi[1, :, :, :]), 0)
				# img2 = torch.cat((img_or2, img_out[1, :, :, :]), 0)
				# viz.images(img1.view(-1, 1, args.Height, args.Width), win='x')
				
				ir_sa, vi_sa, ir_ca, vi_ca, c_fe = middle_temp[0], middle_temp[1], middle_temp[2], middle_temp[3], middle_temp[4]
				img_fe = torch.cat((ir_sa[0, :, :, :], vi_sa[0, :, :, :]), 0)
				img_fe = torch.cat((img_fe, ir_ca[0, :, :, :]), 0)
				img_fe = torch.cat((img_fe, vi_ca[0, :, :, :]), 0)
				img_fe = torch.cat((img_fe, c_fe[0, :, :, :]), 0)
				# viz.images(img_fe.view(-1, 1, args.Height, args.Width), win='y')
				
				weight = torch.cat((weights[0][0, :, :, :], weights[1][0, :, :, :]), 0)
				weight_fuse = torch.cat((weight, weights[2][0, :, :, :]), 0)
				# weight_fuse = torch.cat((weight_fuse, max_temp[0, :, :, :]), 0)
				# viz.images(weight_fuse.view(-1, 1, args.Height, args.Width), win='z')
				# viz.images(weights[3][0, :, :, :].view(-1, 1, args.Height, args.Width), win='z1')
				
				print(mesg)
				loss_p4 = 0.
				loss_p5 = 0.
				loss_p6 = 0.
				loss_p7 = 0.
				loss_p8 = 0.
				loss_p9 = 0.
				loss_p10 = 0.
				loss_all = 0.
		
		# with torch.no_grad():
		# 	print('Start. Testing image data on epoch {}'.format(e + 1))
		# 	test(model_auto_ir, model_auto_vi, model, shift_flag, e + 1)
		# 	print('Done. Testing image data on epoch {}'.format(e + 1))
		
		# save loss
		save_model_filename = 'loss_data_trans_e%d.mat' % (e)
		loss_filename_path = os.path.join(temp_path_loss, save_model_filename)
		scio.savemat(loss_filename_path, {'loss_data': loss_mat})
		# save model
		model.eval()
		model.cpu()
		save_model_filename = "fusetrans_epoch_" + str(e + 1) + ".model"
		save_model_path = os.path.join(temp_path_model, save_model_filename)
		torch.save(model.state_dict(), save_model_path)
		##############
		model.train()
		model.cuda()
		print("\nCheckpoint, trained model saved at: " + save_model_path)
	
	print("\nDone, TransFuse training phase.")


if __name__ == "__main__":
	# True - RGB, False - gray
	if args.channel == 1:
		img_flag = False
	else:
		img_flag = True
	
	path = args.path_ir
	train_num = args.train_num
	data = load_data(path, train_num)
	
	train(data, img_flag)
