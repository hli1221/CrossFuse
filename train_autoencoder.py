# -*- encoding: utf-8 -*-
'''
@Author  :   Hui Li, Jiangnan University
@Contact :   lihui.cv@jiangnan.edu.cn
@File    :   train_autoencoder.py
@Time    :   2024/06/15 16:28:59
'''

# here put the import lib
# Train auto-encoder model for infrarde or visible image

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
from network.net_autoencoder import Auto_Encoder_single

from args_auto import Args as args

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

# -------------------------------------------------------
def load_data(path, train_num):
	# train_num for KAIST
	imgs_path, _ = utils.list_images_datasets(path, train_num)
	# imgs_path = imgs_path[:train_num]
	random.shuffle(imgs_path)
	return imgs_path


# -------------------------------------------------------
def train(data, img_flag):
	batch_size = args.batch
	step = args.step
	
	# auto-encoder
	model = Auto_Encoder_single(**custom_config_auto)
	# model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
	
	if args.resume_model_auto is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_model_auto))
		model.load_state_dict(torch.load(args.resume_model_auto))
	
	# ------------------------------------------------------
	trainable_params = [{'params': filter(lambda x: x.requires_grad, model.parameters()), 'lr': args.lr}]
	optimizer = Adam(trainable_params, args.lr, weight_decay=0.9)

	# visdom
	# viz = Visdom()
	
	if args.cuda:
		model.cuda()

	print('Start training.....')
	
	# creating save path
	temp_path_model = os.path.join(args.save_auto_model)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(temp_path_model, 'loss')
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)
	
	loss_p4 = 0.
	loss_p5 = 0.
	loss_all = 0.
	
	loss_mat = []
	model.train()
	count = 0
	for e in range(args.epochs):
		lr_cur = utils.adjust_learning_rate(optimizer, e, args.lr)
		img_paths, batch_num = utils.load_dataset(data, batch_size)
		
		for idx in range(batch_num):
			
			image_paths = img_paths[idx * batch_size:(idx * batch_size + batch_size)]
			img = utils.get_train_images(image_paths, height=args.Height, width=args.Width, flag=img_flag)
			
			count += 1
			optimizer.zero_grad()
			batch = Variable(img, requires_grad=False)
			
			if args.cuda:
				batch = batch.cuda()
			
			# for DataParallel
			outputs = model.train_module(batch)
			
			img_out = outputs['out']
			recon_loss = outputs['recon_loss']
			ssim_loss = outputs['ssim_loss']
			total_loss = outputs['total_loss']
			loss_mat.append(total_loss.item())
			total_loss.backward()
			optimizer.step()
			
			loss_p4 += outputs['recon_loss']
			loss_p5 += outputs['ssim_loss']
			loss_all += total_loss
			
			# # Test
			# if count % 1000 == 0:
			# 	with torch.no_grad():
			# 		test(model_auto_ir, model_auto_vi, model, e + 1)
			# 		print('Done. Testing image data on epoch {}'.format(e + 1))
			
			if count % step == 0:
				loss_p4 /= step
				loss_p5 /= step
				loss_all /= step
				# if e == 0 and count == step:
				# 	viz.line([loss_all.item()], [0.], win='train_loss', opts=dict(title='Total Loss'))
				
				mesg = "{} - Epoch {}/{} - Batch {}/{} - lr:{:.6f} - recon loss: {:.6f} - ssim loss: {:.6f} - total loss: {:.6f} \n". \
					format(time.ctime(), e + 1, args.epochs, idx + 1, batch_num, lr_cur,
				           loss_p4, loss_p5, loss_all)
				
				# viz.line([loss_all.item()], [count], win='train_loss', update='append')
				# img1 = torch.cat((batch[0, :, :, :], img_out[0, :, :, :]), 0)
				# img_or2 = torch.cat((batch_ir[1, :, :, :], batch_vi[1, :, :, :]), 0)
				# img2 = torch.cat((img_or2, img_out[1, :, :, :]), 0)
				# viz.images(img1.view(-1, 1, args.Height, args.Width), win='x')
				
				# ir_sa, vi_sa, ir_ca, vi_ca, c_fe = middle_temp[0], middle_temp[1], middle_temp[2], middle_temp[3], middle_temp[4]
				# img_fe = torch.cat((ir_sa[0, :, :, :], vi_sa[0, :, :, :]), 0)
				# img_fe = torch.cat((img_fe, ir_ca[0, :, :, :]), 0)
				# img_fe = torch.cat((img_fe, vi_ca[0, :, :, :]), 0)
				# img_fe = torch.cat((img_fe, c_fe[0, :, :, :]), 0)
				# viz.images(img_fe.view(-1, 1, args.Height, args.Width), win='y')
				
				# weight = torch.cat((weights[0][0, :, :, :], weights[1][0, :, :, :]), 0)
				# weight_fuse = torch.cat((weight, weights[2][0, :, :, :]), 0)
				# weight_fuse = torch.cat((weight_fuse, max_temp[0, :, :, :]), 0)
				# viz.images(weight_fuse.view(-1, 1, args.Height, args.Width), win='z')
				# viz.images(weights[3][0, :, :, :].view(-1, 1, args.Height, args.Width), win='z1')
				
				print(mesg)
				loss_p4 = 0.
				loss_p5 = 0.
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
		save_model_filename = "auto_encoder_epoch_" + str(e + 1) + "_" + args.type_flag + ".model"
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
	
	path = args.path
	train_num = args.train_num
	data = load_data(path, train_num)
	
	train(data, img_flag)

