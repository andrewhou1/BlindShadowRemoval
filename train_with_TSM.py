import cv2
import tensorflow as tf

import os
import time
import glob
import numpy as np
from tensorflow.keras import layers
from model_with_TSM import Generator, Discriminator
from dataset_with_TSM import Dataset
from utils import l1_loss, l1_loss_hsv, l1_loss_yuv, l2_loss, l2_loss_yuv, hinge_loss, style_content_loss, find_edge, Logging, apply_ss_shadow_map, get_brightness_mask, render_perlin_mask
from warp import tf_batch_map_offsets
from sklearn import metrics
import scipy.io


# Base Configuration Class
class Config(object):
	GPU_INDEX = 0
	DATA_DIR = ['/research/cvlshare/cvl-liuyaoj1/Data/Helen/bin/*',
				'/research/cvlshare/cvl-liuyaoj1/Data/FFHQ/*',]  
	#DATA_DIR_VAL = ['/research/cvl-liuyaoj1/Data/UCB/train/input/*',
	#				'/research/cvl-liuyaoj1/Data/SFW/*',]
	DATA_DIR_VAL = ['/research/cvlshare/cvl-liuyaoj1/Data/UCB/train/input/*']
	DATA_DIR_TEST = ['SFW/*']
	#DATA_DIR_TEST = ['/research/cvlshare/cvl-liuyaoj1/Data/UCB/train/input/*']
	#DATA_DIR_TEST = ['sample_imgs/*']
	LOG_DEVICE_PLACEMENT = False
	IMG_SIZE = 256
	MAP_SIZE = 32
	FIG_SIZE = 128
	# Training Meta
	STEPS_PER_EPOCH = 2000
	#MAX_EPOCH = 60
	#MAX_EPOCH = 100
	MAX_EPOCH = 300
	IMG_LOG_FR = 100
	TXT_LOG_FR = 1000
	NUM_EPOCHS_PER_DECAY = 10.0   # Epochs after which learning rate decays
	#NUM_EPOCHS_PER_DECAY = 25.0
	BATCH_SIZE = 1
	#BATCH_SIZE = 5
	#BATCH_SIZE=7
	#BATCH_SIZE = 3
	LEARNING_RATE = 1e-4          # Initial learning rate.
	LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
	LEARNING_MOMENTUM = 0.999     # The decay to use for the moving average.
	MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

	# Network
	n_layer_D = 4

	def __init__(self, gpu_idx=None):
		if gpu_idx is None:
			gpu_idx = self.GPU_INDEX
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			try:
				tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
				tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
			except RuntimeError as e:
				print(e) # Virtual devices must be set before GPUs have been initialized
		#self.compile()

	def compile(self):
		if not os.path.isdir(self.CHECKPOINT_DIR):
			os.mkdir(self.CHECKPOINT_DIR)
		if not os.path.isdir(self.CHECKPOINT_DIR+'/test'):
			os.mkdir(self.CHECKPOINT_DIR+'/test')
		"""Display Configuration values."""
		print("\nConfigurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)) and a[0].isupper():
				print("{:30} {}".format(a, getattr(self, a)))
		print("\n")

def process_mask(mask, imsize, gt, img_dark, uv, face):
  img_list = []
  mask_sv_list = []
  mask_edge_list = []
  for _mask,_gt,_img_dark,_uv,_face in zip(tf.unstack(mask,axis=0), tf.unstack(gt,axis=0), tf.unstack(img_dark,axis=0), tf.unstack(uv,axis=0), tf.unstack(face,axis=0)):
      _mask = tf.cond(tf.greater(tf.random.uniform([]), .4),
                                lambda: _mask,
                                lambda: _face*render_perlin_mask(size=(imsize, imsize)))
      _mask_ss = tf.cond(tf.greater(tf.random.uniform([]), .25),
                         lambda: apply_ss_shadow_map(1-_mask),
                         lambda: tf.image.grayscale_to_rgb(1-_mask))
      _mask_sv = 1 - _mask_ss
      intensity_mask = tf.cond(tf.greater(tf.random.uniform([]), tf.constant(0.5)),
                         lambda: get_brightness_mask(size=(imsize,imsize), min_val=0.3),
                         lambda: get_brightness_mask(size=(imsize,imsize), min_val=0.5))
      '''intensity_mask = tf.cond(tf.greater(tf.random.uniform([]), tf.constant(0.5)),
                         lambda: get_brightness_mask(size=(imsize,imsize), min_val=0.2),
                         lambda: get_brightness_mask(size=(imsize,imsize), min_val=0.4))'''
      intensity_mask = tf.expand_dims(intensity_mask, 2)
      _img = _gt * _mask_ss + _img_dark * _mask_sv * intensity_mask
      img_list  += [tf.clip_by_value(_img,0,1)]
      mask_sv_list += [tf.reshape(_mask_sv, (imsize, imsize, 3))]
      mask_edge_list += [tf.reshape(tf.abs(_mask_sv-_mask), (imsize, imsize, 3))]
      #inten_inv    += [1 / (intensity_mask+1e-6)]
  return tf.stack(img_list, axis=0), tf.stack(mask_sv_list, axis=0) , tf.stack(mask_edge_list, axis=0)

def get_img_grad(img, scale=1):
  b, w, h, c = img.shape
  if scale > 1:
     img = tf.image.resize(img, [w//scale, h//scale])
  grad_x, grad_y = tf.image.image_gradients(img)
  grad = (grad_x + grad_y) * 5
  if scale > 1:
  	 grad = tf.image.resize(grad, [w, h])
  return grad

class FSRNet(object):
	def __init__(self, config):
		self.config = config
		self.gen = Generator()
		self.disc1 = Discriminator(1,config.n_layer_D)
		self.disc2 = Discriminator(2,config.n_layer_D)
		self.disc3 = Discriminator(4,config.n_layer_D) 
		self.gen_opt = tf.keras.optimizers.Adam(config.LEARNING_RATE)
		self.disc_opt = tf.keras.optimizers.Adam(config.LEARNING_RATE)

		# perceptual loss
		self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
		self.vgg_style_layers = ['block1_conv1',
				                 'block2_conv1',
				                 'block3_conv1', 
				                 'block4_conv1', 
				                 'block5_conv1']
		'''self.vgg_style_layers = ['block1_conv2',
				                 'block2_conv2',
				                 'block3_conv2', 
				                 'block4_conv2', 
				                 'block5_conv2']'''
		self.feat_extractor = self.vgg_feat_extractor()

		# checkpoint
		self.checkpoint_prefix = os.path.join(config.CHECKPOINT_DIR, "ckpt")
		self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_opt,
										 	  discriminator_optimizer=self.disc_opt,
										 	  generator=self.gen,
										 	  discriminator_1=self.disc1,
										 	  discriminator_2=self.disc2,
										 	  discriminator_3=self.disc3)

		# losses
		self.log = Logging(config)

	def vgg_feat_extractor(self):
		vgg = self.vgg
		vgg_layers = self.vgg_style_layers
		vgg.trainable = False
		outputs = [vgg.get_layer(name).output for name in vgg_layers]
		feat_extractor = tf.keras.Model([vgg.input], outputs)
        # input * 255
		return feat_extractor

	def update_lr(self, lr):
		self.gen_opt = tf.keras.optimizers.Adam(lr)
		self.disc_opt = tf.keras.optimizers.Adam(lr)	

	def train(self, dataset, dataset_val):
		# find restore
		last_checkpoint = tf.train.latest_checkpoint(self.config.CHECKPOINT_DIR)
		if last_checkpoint:
			last_epoch = int(last_checkpoint.split('-')[-1])
			self.checkpoint.restore(last_checkpoint)
		else:
			last_epoch = 0
		print('**********************************************************')
		print('Restore from Epoch '+str(last_epoch))
		print('**********************************************************')

		for epoch in range(last_epoch, self.config.MAX_EPOCH):
			start = time.time()
			training=True
			for step in range(self.config.STEPS_PER_EPOCH):
				img_batch, name_batch = next(dataset.feed)
				losses, figs = self.train_step(img_batch, training)
				self.log.display(losses, epoch, step, training, self.config.STEPS_PER_EPOCH)
				self.log.save(figs, training)

			self.checkpoint.save(file_prefix = self.checkpoint_prefix)
			print('')

			training=False
			for step in range(self.config.STEPS_PER_EPOCH//10):
				img_batch, clr_batch = next(dataset_val.feed)
				losses, figs = self.train_step(img_batch, training)
				self.log.display(losses, epoch, step, training, self.config.STEPS_PER_EPOCH//10)
				self.log.save(figs, training)

			print ('\n*****Time for epoch {} is {} sec*****'.format(epoch + 1, int(time.time()-start)))
	
	'''
	img_m = tf.image.flip_left_right(img)
	mask_sv_m = tf.image.flip_left_right(mask_sv)
	gt_m = tf.image.flip_left_right(gt)
	# concat
	img = tf.concat([img,img_m],axis=0)
	gt = tf.concat([gt,gt_m],axis=0)
	mask_sv = tf.concat([mask_sv,mask_sv_m],axis=0)
	uv = tf.concat([uv,uv_m],axis=0)
	'''

	# Notice the use of `tf.function`
	# This annotation causes the function to be "compiled".
	@tf.function
	def train_step(self, img, training):
		losses = {}
		figs = []
		img = tf.reshape(img, [self.config.BATCH_SIZE*2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
		img_dark, gt, mask, uv, reg, face = tf.split(img, [3, 3, 1, 3, 6, 1], 3)
		frame = 1

		# data aug
		if training:
			img_dark_pack = []
			gt_pack = []
			gt = tf.reshape(gt, [self.config.BATCH_SIZE, 2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
			img_dark = tf.reshape(img_dark, [self.config.BATCH_SIZE, 2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
			for _img_dark, _gt in zip(tf.unstack(img_dark,axis=0), tf.unstack(gt,axis=0)):
				rd = tf.random.uniform([])
				_gt = tf.cond(tf.greater(rd, .5),
							lambda: _gt,
							lambda: tf.image.random_saturation(_gt, 0.5, 2))
				_img_dark = tf.cond(tf.greater(rd, .5),
							lambda: _img_dark,
							lambda: tf.image.random_saturation(_img_dark, 0.5, 2))
				img_dark_pack.append(_img_dark)
				gt_pack.append(_gt)
			img_dark = tf.stack(img_dark_pack, axis=0)
			gt = tf.stack(gt_pack, axis=0)
			img_dark = tf.reshape(img_dark, [self.config.BATCH_SIZE*2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
			gt = tf.reshape(gt, [self.config.BATCH_SIZE*2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
		img, mask_sv, mask_edge = process_mask(mask, self.config.IMG_SIZE, gt, img_dark, uv, face)
		if training:
			img_dark_pack = []
			gt_pack = []
			img_0 = tf.reshape(img, [self.config.BATCH_SIZE, 2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
			img_l, _ = tf.split(img_0, 2, 1)
			img_r = tf.image.flip_left_right(img_l[:,0,:,:,:])
			img_0 = tf.stack([img_l[:,0,:,:,:], img_r], axis=1)
			img_0 = tf.reshape(img_0, [self.config.BATCH_SIZE*2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
			img = tf.cond(tf.greater(tf.random.uniform([]), .35),
							lambda: img_0,
							lambda: img)
		else:
			img = gt

		if training:
			share = tf.greater(tf.random.uniform([]), 0.5)
		else:
			share = tf.greater(tf.random.uniform([]), 0.)
			img = gt

		# mask
		mask_bi = tf.cast(tf.greater(mask_sv, .01), tf.float32)
		mask_edge = find_edge(mask_sv) #tf.cast(tf.greater(tf.reduce_mean(mask_edge,axis=3,keepdims=True), .01), tf.float32)
		#dif = tf.image.rgb_to_grayscale(tf.reverse(gt, axis=[-1])) - tf.image.rgb_to_grayscale(tf.reverse(img, axis=[-1]))
		dif = tf.image.rgb_to_grayscale(gt) - tf.image.rgb_to_grayscale(img)
		bmaskgt = tf.cast(tf.greater(dif, 0.04),tf.float32)

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			deshadow_img_gs, deshadow_img_c, mask_pred, bmask= self.gen(img, uv, reg, frame, share, chuck=2, training=training)

			#print(tf.print(deshadow_img_gs))
			#print(tf.print(deshadow_img_c))
			#print(tf.print(gt))

			d_img = tf.concat([gt,deshadow_img_c], axis=0)  
			d_mask = tf.concat([mask_sv,mask_sv], axis=0)  
			d_output_1 = self.disc1(tf.concat([d_img,d_mask], axis=3), training=training)
			d_output_2 = self.disc2(tf.concat([d_img,d_mask], axis=3), training=training)
			d_output_3 = self.disc3(tf.concat([d_img,d_mask], axis=3), training=training)
			#losses
			mask_loss = l1_loss(mask_pred, mask_bi)
			'''recon_loss_gs = (l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(tf.reverse(gt, axis=[-1]))) + \
							 l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(tf.reverse(gt, axis=[-1])), mask_bi)*30+  \
							 l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(tf.reverse(gt, axis=[-1])), mask_edge)*10)/41
			recon_loss_c =  (l1_loss(deshadow_img_c, gt)+l1_loss(deshadow_img_c, gt, mask_bi)*30+l1_loss(deshadow_img_c, gt, mask_edge)*10+\
							 l1_loss_yuv(deshadow_img_c, gt)+l1_loss_yuv(deshadow_img_c, gt, mask_bi)*30+l1_loss_yuv(deshadow_img_c, gt, mask_edge)*10)/82'''
			'''recon_loss_gs = (l2_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(tf.reverse(gt, axis=[-1]))) + \
							 l2_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(tf.reverse(gt, axis=[-1])), mask_bi)*30+  \

							 l2_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(tf.reverse(gt, axis=[-1])), mask_edge)*10)/41'''
			'''recon_loss_gs = (l2_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt)) + \
							 l2_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt), mask_bi)*30+  \
							 l2_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt), mask_edge)*10)/41

			recon_loss_c =  (l2_loss(deshadow_img_c, gt)+l2_loss(deshadow_img_c, gt, mask_bi)*30+l2_loss(deshadow_img_c, gt, mask_edge)*10+\

							 l2_loss_yuv(deshadow_img_c, gt)+l2_loss_yuv(deshadow_img_c, gt, mask_bi)*30+l2_loss_yuv(deshadow_img_c, gt, mask_edge)*10)/82'''
			recon_loss_gs = (l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt)) + \
							 l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt), mask_bi)*30+  \
							 l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt), mask_edge)*10)/41

			recon_loss_c =  (l1_loss(deshadow_img_c, gt)+l1_loss(deshadow_img_c, gt, mask_bi)*30+l1_loss(deshadow_img_c, gt, mask_edge)*10+\

							 l1_loss_yuv(deshadow_img_c, gt)+l1_loss_yuv(deshadow_img_c, gt, mask_bi)*30+l1_loss_yuv(deshadow_img_c, gt, mask_edge)*10)/82
			'''recon_loss_gs = (l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt))*10 + \
							 l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt), mask_bi)*30+  \
							 l1_loss(deshadow_img_gs, tf.image.rgb_to_grayscale(gt), mask_edge)*10)/51

			recon_loss_c =  (l1_loss(deshadow_img_c, gt)*10+l1_loss(deshadow_img_c, gt, mask_bi)*30+l1_loss(deshadow_img_c, gt, mask_edge)*10+\

							 l1_loss_yuv(deshadow_img_c, gt)*10+l1_loss_yuv(deshadow_img_c, gt, mask_bi)*30+l1_loss_yuv(deshadow_img_c, gt, mask_edge)*10)/102'''
			recon_loss = (recon_loss_gs + recon_loss_c) / 2
			gan_loss   = - tf.reduce_mean(d_output_1[1]) - tf.reduce_mean(d_output_2[1]) - tf.reduce_mean(d_output_3[1])
			per_loss = style_content_loss(self.feat_extractor, d_img)

			#DSSIM_loss = 1-tf.reduce_mean(tf.image.ssim(gt, deshadow_img_c, max_val=1.0))

			grad_gt_1 = get_img_grad(gt, scale = 1) 
			grad_gt_2 = get_img_grad(gt, scale = 2)
			grad_gt_3 = get_img_grad(gt, scale = 4)
			grad_gt_4 = get_img_grad(gt, scale = 8)
			grad_gt_5 = get_img_grad(gt, scale = 16)
			grad_rc_1 = get_img_grad(deshadow_img_c, scale = 1) 
			grad_rc_2 = get_img_grad(deshadow_img_c, scale = 2)
			grad_rc_3 = get_img_grad(deshadow_img_c, scale = 4)
			grad_rc_4 = get_img_grad(deshadow_img_c, scale = 8)
			grad_rc_5 = get_img_grad(deshadow_img_c, scale = 16)
			'''dif_grad_1 = tf.abs(grad_rc_1 - grad_gt_1)
			dif_grad_2 = tf.abs(grad_rc_2 - grad_gt_2)
			dif_grad_3 = tf.abs(grad_rc_3 - grad_gt_3)
			dif_grad_4 = tf.abs(grad_rc_4 - grad_gt_4)
			dif_grad_5 = tf.abs(grad_rc_5 - grad_gt_5)'''
			#reweight shadow mask gradients
			dif_grad_1 = (tf.abs(grad_rc_1 - grad_gt_1)+30*tf.abs(grad_rc_1 - grad_gt_1)*mask_bi+10*tf.abs(grad_rc_1 - grad_gt_1)*mask_edge)/41
			dif_grad_2 = (tf.abs(grad_rc_2 - grad_gt_2)+30*tf.abs(grad_rc_2 - grad_gt_2)*mask_bi+10*tf.abs(grad_rc_2 - grad_gt_2)*mask_edge)/41
			dif_grad_3 = (tf.abs(grad_rc_3 - grad_gt_3)+30*tf.abs(grad_rc_3 - grad_gt_3)*mask_bi+10*tf.abs(grad_rc_3 - grad_gt_3)*mask_edge)/41
			dif_grad_4 = (tf.abs(grad_rc_4 - grad_gt_4)+30*tf.abs(grad_rc_4 - grad_gt_4)*mask_bi+10*tf.abs(grad_rc_4 - grad_gt_4)*mask_edge)/41
			dif_grad_5 = (tf.abs(grad_rc_5 - grad_gt_5)+30*tf.abs(grad_rc_5 - grad_gt_5)*mask_bi+10*tf.abs(grad_rc_5 - grad_gt_5)*mask_edge)/41
			grad_loss = tf.reduce_sum(dif_grad_1 + dif_grad_2 + dif_grad_3 + dif_grad_4 + dif_grad_5) / (tf.reduce_sum(mask_edge)+1e-6)
			g_total_loss = recon_loss*400+ gan_loss + per_loss * .005 + grad_loss*2
			#g_total_loss = recon_loss*400+ gan_loss + per_loss * .005 + grad_loss*2+DSSIM_loss*100
			#g_total_loss = recon_loss*2000+ gan_loss + per_loss * .005 + grad_loss
			#g_total_loss = recon_loss*400+ gan_loss/5 + per_loss * .005 + grad_loss/10
			# d losses
			d_loss_r = hinge_loss(d_output_1[0], 1) + hinge_loss(d_output_2[0], 1) + hinge_loss(d_output_3[0], 1) 
			d_loss_s = hinge_loss(d_output_1[1],-1) + hinge_loss(d_output_2[1],-1) + hinge_loss(d_output_3[1],-1) 
			d_total_loss =  d_loss_r + d_loss_s 
		
		if training:
			gen_trainable_vars = self.gen.trainable_variables
			disc_trainable_vars = self.disc1.trainable_variables +\
						          self.disc2.trainable_variables +\
						          self.disc3.trainable_variables

			g_gradients = gen_tape.gradient(g_total_loss, gen_trainable_vars)
			d_gradients = disc_tape.gradient(d_total_loss, disc_trainable_vars)

			self.gen_opt.apply_gradients(zip(g_gradients, gen_trainable_vars))
			self.disc_opt.apply_gradients(zip(d_gradients, disc_trainable_vars))
		losses['recon_gs'] = recon_loss_gs
		losses['recon_c'] = recon_loss_c
		#losses['DSSIM'] = DSSIM_loss
		losses['grad'] = grad_loss
		losses['gen'] = gan_loss
		losses['disc_real'] = d_loss_r
		losses['disc_fake'] = d_loss_s

		figs = [img, gt, deshadow_img_c, deshadow_img_gs, mask_edge, bmaskgt, bmask, (dif_grad_1+dif_grad_2+dif_grad_3+dif_grad_4+dif_grad_5)/1.2]
		return losses, figs

	def test(self, dataset_val):
		# find restore
		last_checkpoint = tf.train.latest_checkpoint(self.config.CHECKPOINT_DIR)
		if last_checkpoint:
			last_epoch = int(last_checkpoint.split('-')[-1])
			self.checkpoint.restore(last_checkpoint).expect_partial()
		else:
			last_epoch = 0
		print('**********************************************************')
		print('Restore from Epoch '+str(last_epoch))
		print('**********************************************************')

		masks = sorted(os.listdir('../../../UCB_input_images_face_masks_cropped_and_padded_with_hair/'))
		#masks = sorted(os.listdir('../../../UCB_input_images_face_masks_cropped_and_padded/'))
	
		start = time.time()
		num_list = len(dataset_val.name_list)
		#print(dataset_val.name_list)
		print(masks)
		count = 0
		frac_in_nose_arr = np.zeros((100))
		mean_intensity = np.zeros((100))
		for step, img_name in enumerate(dataset_val.name_list): #range(num_list):
			#print(enumerate(dataset_val.name_list))
			img, box, _ = next(dataset_val.feed)
			curr_mask = cv2.imread('../../../UCB_input_images_face_masks_cropped_and_padded_with_hair/'+masks[count])/255.0
			curr_mask_no_hair = cv2.imread('../../../UCB_input_images_face_masks_cropped_and_padded/'+masks[count])/255.0
			curr_mouth_mask = cv2.imread('../../../UCB_input_images_mouth_masks_cropped_and_padded/'+masks[count])/255.0
			curr_nose_mask = cv2.imread('../../../UCB_input_images_nose_masks_cropped_and_padded/'+masks[count])/255.0
			curr_eyebrow_mask = cv2.imread('../../../UCB_input_images_eyebrow_masks_cropped_and_padded/'+masks[count])/255.0
			curr_eye_mask = cv2.imread('../../../UCB_input_images_eye_masks_cropped_and_padded/'+masks[count])/255.0
			curr_glasses_mask = cv2.imread('../../../UCB_input_images_glasses_masks_cropped_and_padded/'+masks[count])/255.0
			count += 1
			curr_mask = tf.convert_to_tensor(curr_mask)
			curr_mask_no_hair = tf.convert_to_tensor(curr_mask_no_hair)
			curr_mouth_mask = tf.convert_to_tensor(curr_mouth_mask)
			curr_nose_mask = tf.convert_to_tensor(curr_nose_mask)
			curr_eyebrow_mask = tf.convert_to_tensor(curr_eyebrow_mask)
			curr_eye_mask = tf.convert_to_tensor(curr_eye_mask)
			curr_glasses_mask = tf.convert_to_tensor(curr_glasses_mask)
			losses, figs, frac_in_nose_arr[count-1], mean_intensity[count-1] = self.test_step(img, box, curr_mask, curr_mask_no_hair, curr_mouth_mask, curr_nose_mask, curr_eyebrow_mask, curr_eye_mask, curr_glasses_mask, training=False)
			self.log.display(losses, 0, step, False, num_list)
			self.log.save_img(figs, img_name)

		print ('\n*****Time for epoch {} is {} sec*****'.format(1, int(time.time()-start)))
		output_mat = {}
		output_mat['frac_in_nose'] = frac_in_nose_arr
		output_mat['mean_intensity'] = mean_intensity
		scipy.io.savemat('frac_in_nose.mat', output_mat)

	#@tf.function
	def test_step(self, img, box, curr_mask, curr_mask_no_hair, curr_mouth_mask, curr_nose_mask, curr_eyebrow_mask, curr_eye_mask, curr_glasses_mask, training):
		losses = {}
		figs = []
		#print(tf.shape(img))
		img = tf.reshape(img, [2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
		box = tf.reshape(box, [4])
		size = box[3] - box[1]

		img, gt, uv, reg, face = tf.split(img, [3, 3, 3, 6, 1], 3)
		#print(face[1, :, :, :] == face[2, :, :, :])
		#print(tf.reduce_sum(tf.cast(img[1, :, :, :] == img[2, :, :, :], tf.float32)))
		frame = 1
		share = tf.greater(tf.random.uniform([]), 0.)
		deshadow_img_gs, deshadow_img_c, _, mask_pred = self.gen(img, uv, reg, frame, share, chuck=4, training=training)

		#print(tf.reduce_max(face))
		#print(tf.reduce_min(face))

		#print(deshadow_img_c)

		#mask_pred = mask_pred * face
		#mask_pred = mask_pred * face
		#deshadow_img_c = deshadow_img_c * face + img*(1-face)
		#deshadow_img_c = deshadow_img_c
		#deshadow_img_c_sc = tf.math.maximum(deshadow_img_c[0,...], tf.image.flip_left_right(deshadow_img_c[1,...]))
		#deshadow_img_c = tf.clip_by_value(deshadow_img_c, 0, 1)
		gt_sc = gt[0,...]
		#deshadow_img_c_sc = deshadow_img_c[0,...]

		gt_sc = tf.image.resize(gt_sc, [size,size])
		#deshadow_img_c_sc_orig = tf.image.resize(deshadow_img_c[0,...], [size,size])
		#deshadow_img_c_sc_flipped = tf.image.resize(deshadow_img_c[1,...], [size,size])
		deshadow_img_c_sc_orig = deshadow_img_c[0,...]
		deshadow_img_c_sc_flipped = deshadow_img_c[1,...]
		orig_mask = curr_mask
		'''curr_mask = tf.image.resize(curr_mask, [size,size])
		curr_mask = tf.round(curr_mask)
		curr_mask_no_hair = tf.image.resize(curr_mask_no_hair, [size,size])
		curr_mask_no_hair = tf.round(curr_mask_no_hair)
		curr_nose_mask = tf.image.resize(curr_nose_mask, [size,size])
		curr_nose_mask = tf.round(curr_nose_mask)
		curr_mouth_mask = tf.image.resize(curr_mouth_mask, [size,size])
		curr_mouth_mask = tf.round(curr_mouth_mask)
		curr_eyebrow_mask = tf.image.resize(curr_eyebrow_mask, [size,size])
		curr_eyebrow_mask = tf.round(curr_eyebrow_mask)
		curr_eye_mask = tf.image.resize(curr_eye_mask, [size,size])
		curr_eye_mask = tf.round(curr_eye_mask)
		curr_glasses_mask = tf.image.resize(curr_glasses_mask, [size,size])
		curr_glasses_mask = tf.round(curr_glasses_mask)'''
		#print(gt_sc.shape, deshadow_img_c_sc.shape)
		gt_sc = tf.pad(gt_sc, [[0,256-size],[0,256-size],[0,0]])

		tmp = img[0,...]
		#tmp = tf.image.resize(tmp, [size,size])
		#tmp = tf.pad(tmp, [[0,256-size],[0,256-size],[0,0]])
		
		#print(tf.reduce_max(curr_mask))
		#print(tf.reduce_min(curr_mask))
		#print(tf.reduce_sum(curr_mask))

		'''curr_mask = tf.cast(tf.pad(curr_mask, [[0,256-size],[0,256-size],[0,0]]), tf.float32)
		curr_mask_no_hair = tf.cast(tf.pad(curr_mask_no_hair, [[0,256-size],[0,256-size],[0,0]]), tf.float32)
		curr_nose_mask = tf.cast(tf.pad(curr_nose_mask, [[0,256-size],[0,256-size],[0,0]]), tf.float32)
		curr_mouth_mask = tf.cast(tf.pad(curr_mouth_mask, [[0,256-size],[0,256-size],[0,0]]), tf.float32)
		curr_eyebrow_mask = tf.cast(tf.pad(curr_eyebrow_mask, [[0,256-size],[0,256-size],[0,0]]), tf.float32)
		curr_eye_mask = tf.cast(tf.pad(curr_eye_mask, [[0,256-size],[0,256-size],[0,0]]), tf.float32)
		curr_glasses_mask = tf.cast(tf.pad(curr_glasses_mask, [[0,256-size],[0,256-size],[0,0]]), tf.float32)'''

		mask_pred = mask_pred[0,...]
		#print(tf.reduce_max(mask_pred))
		'''mask_pred = tf.image.resize(mask_pred, [size,size])
		mask_pred = tf.pad(mask_pred, [[0,256-size],[0,256-size],[0,0]])
		mask_pred = mask_pred * tf.cast(curr_mask, tf.float32)'''
		mask_pred = mask_pred * tf.cast(orig_mask, tf.float32)

		hair_region = tf.cast(curr_mask-curr_mask_no_hair, tf.float32)
		threshold = np.zeros((256, 256, 3))
		threshold = threshold+0.01
		'''img_intensity = tf.reshape(tf.reduce_mean(tmp, 2), (256, 256, 1))
		img_intensity = tf.concat([img_intensity, img_intensity, img_intensity], 2)
		#threshold[tf.logical_and(hair_region > 0, img_intensity < 0.25)] = 0.005
		threshold[hair_region > 0] = 0.02
		threshold[tf.logical_and(hair_region > 0, img_intensity < 0.13)] = 0.004
		#threshold[tf.logical_and(tf.convert_to_tensor(np.reshape(curr_mask_no_hair[:, :, 0], (256, 256, 1))) > 0, img_intensity > 0.8)] = 1.0

		#handle forehead
		if(np.sum(curr_eyebrow_mask) > 30):
			forehead_mask = curr_mask_no_hair.numpy()
			(rows, cols) = np.where(curr_eyebrow_mask[:, :, 0] == 1)
			upper_brow = np.min(rows)
			forehead_mask[upper_brow:256, :, :] = 0
			(rows, cols) = np.where(forehead_mask[:, :, 0] == 1)
			upper_forehead = np.min(rows)
			left = np.min(cols)
			right = np.max(cols)
			forehead_mask = np.zeros((256, 256, 3))
			forehead_mask[int(upper_forehead+20):int(upper_brow-40), int(left+40):int(right-40)] = 1
			threshold[tf.logical_and(tf.convert_to_tensor(forehead_mask) > 0, img_intensity < 0.4)] = -0.001'''

		#detected_shadow_mask = tf.cast(tf.cast(mask_pred > 0.01, tf.uint8), tf.float32)
		detected_shadow_mask = tf.cast(tf.greater(tf.cast(mask_pred, tf.float32), tf.cast(tf.convert_to_tensor(threshold), tf.float32)), tf.float32)

		detected_shadow_mask = detected_shadow_mask.numpy()
		detected_shadow_mask = detected_shadow_mask.astype(np.uint8)
		#print(np.shape(detected_shadow_mask))
		#detected_shadow_mask = cv2.medianBlur(detected_shadow_mask, 11)
		#find all your connected components (white blobs in your image)
		nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(detected_shadow_mask[:, :, 0], connectivity=4)
		#connectedComponentswithStats yields every seperated component with information on each of them, such as size
		#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
		sizes = stats[1:, -1]; nb_components = nb_components - 1
		#print(sizes)

		# minimum size of particles we want to keep (number of pixels)
		#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
		#min_size = min(2000, 0.45*np.max(sizes))
		min_size = 0.6*np.max(sizes)

		#your answer image
		img2 = np.zeros((256, 256, 1))
		#for every component in the image, you keep it only if it's above min_size
		for i in range(0, nb_components):
			tmp_img = np.zeros((256, 256, 1))
			tmp_img[output == i+1] = 1
			test_if_hair_only = np.reshape(hair_region.numpy()[:, :, 0], (256, 256, 1))*tmp_img
			if ((sizes[i] >= min_size) and (np.sum(test_if_hair_only)/sizes[i] < 0.8)):
        			img2[output == i + 1] = 1

		#handle nose
		shadow_image = img2*np.reshape(np.mean(tmp.numpy(), 2), (256, 256, 1))
		mean_intensity = np.sum(shadow_image)/np.sum(img2)
		nose_image = tf.cast(curr_nose_mask, tf.float32)*tmp
		frac_nose_in_shadow = np.sum((np.reshape(curr_nose_mask[:, :, 0], (256, 256, 1))*shadow_image) > 0)/np.sum(curr_nose_mask[:, :, 0])
		#print(curr_nose_mask[:, :, 0]*shadow_image > 0)
		#print(np.sum(curr_nose_mask[:, :, 0]*shadow_image > 0))
		#print(np.sum(curr_nose_mask[:, :, 0]))
		print(frac_nose_in_shadow)
		(rows, cols) = np.where(curr_nose_mask[:, :, 0] == 1)
		mid_nose_height = (np.max(rows)+np.min(rows))/2.0
		lower_nose = np.max(rows)
		mid_nose_width = (np.max(cols)+np.min(cols))/2.0
		if((frac_nose_in_shadow > 0.423 and frac_nose_in_shadow < 0.425) or (frac_nose_in_shadow > 0.53 and frac_nose_in_shadow < 0.56) or (frac_nose_in_shadow > 0.35 and frac_nose_in_shadow < 0.38) or (frac_nose_in_shadow > 0.58 and frac_nose_in_shadow < 0.605)):
			if(mean_intensity < 0.15):
				img2[int(mid_nose_height):int(lower_nose+5), int(mid_nose_width-35):int(mid_nose_width+35)] = 0
			else:
				img2[int(mid_nose_height):int(lower_nose+65), int(mid_nose_width-35):int(mid_nose_width+35)] = 0

		detected_shadow_mask = np.concatenate((img2, img2, img2), axis=2)
		#detected_shadow_mask = cv2.GaussianBlur(detected_shadow_mask,(11,11),cv2.BORDER_DEFAULT)

		detected_shadow_mask = tf.cast(tf.convert_to_tensor(detected_shadow_mask), tf.float32)

		#detected_shadow_mask = tf.concat([detected_shadow_mask, detected_shadow_mask, detected_shadow_mask], 2)
		#deshadow_img_c_sc_orig = tf.pad(deshadow_img_c_sc_orig, [[0,256-size],[0,256-size],[0,0]])
		#deshadow_img_c_sc_flipped = tf.pad(deshadow_img_c_sc_flipped, [[0,256-size],[0,256-size],[0,0]])
		#check_full_pred = deshadow_img_c_sc
		#deshadow_img_c_sc = tf.cast(deshadow_img_c_sc, tf.float32)*tf.cast(curr_mask, tf.float32)+tf.cast(tmp, tf.float32)*tf.cast((1-curr_mask), tf.float32)
		#deshadow_img_c_sc = tf.math.maximum(deshadow_img_c_sc_orig, tf.image.flip_left_right(deshadow_img_c_sc_flipped))
		#deshadow_img_c_sc = tf.cast(deshadow_img_c_sc, tf.float32)*tf.cast(detected_shadow_mask, tf.float32)+tf.cast(tmp, tf.float32)*tf.cast((1-detected_shadow_mask), tf.float32)
		deshadow_img_c_sc_orig = tf.cast(deshadow_img_c_sc_orig, tf.float32)*tf.cast(detected_shadow_mask, tf.float32)+tf.cast(tmp, tf.float32)*tf.cast((1-detected_shadow_mask), tf.float32)
		deshadow_img_c_sc_flipped = tf.cast(deshadow_img_c_sc_flipped, tf.float32)*tf.cast(tf.image.flip_left_right(detected_shadow_mask), tf.float32)+tf.cast(tf.image.flip_left_right(tmp), tf.float32)*tf.cast(tf.image.flip_left_right(1-detected_shadow_mask), tf.float32)
		#deshadow_img_c_sc = tf.cast(deshadow_img_c_sc, tf.float32)*tf.cast(curr_mask, tf.float32)+tf.cast(img[0,...], tf.float32)*tf.cast((1-curr_mask), tf.float32)
		#curr_mask = tf.cast(tf.pad(curr_mask, [[0,256-size],[0,256-size],[0,0]]), tf.float32)
		#print(tf.shape(deshadow_img_c_sc))
		#print(tf.reduce_max(deshadow_img_c_sc))
		#print(tf.reduce_min(deshadow_img_c_sc))
		#print(deshadow_img_c_sc)
		#deshadow_img_c_sc = tf.math.maximum(deshadow_img_c_sc_orig, tf.image.flip_left_right(deshadow_img_c_sc_flipped))
		deshadow_img_c_sc = tf.clip_by_value(deshadow_img_c_sc_orig, 0, 1)
		deshadow_img_c_sc = tf.image.resize(deshadow_img_c_sc, [size,size])
		deshadow_img_c_sc = tf.pad(deshadow_img_c_sc, [[0,256-size],[0,256-size],[0,0]])
           
		#print(gt_sc.shape, deshadow_img_c_sc.shape)
		#input()
		#losses
		#print(tf.shape(deshadow_img_c_sc))
		ssim = tf.reduce_sum(tf.image.ssim(gt_sc, deshadow_img_c_sc, max_val=1.0))
		psnr = tf.reduce_sum(tf.image.psnr(gt_sc, deshadow_img_c_sc, max_val=1.0))
		
		losses['ssim'] = ssim
		losses['psnr'] = psnr
		#deshadow_img_c = tf.stack([deshadow_img_c_sc, tf.image.flip_left_right(deshadow_img_c_sc)],axis=0)
		#mask_progress = tf.image.resize(mask_progress, [256,256])
		#figs = [img, deshadow_img_c, mask_pred*2]
		#print(tf.shape(tmp))
		'''print(tf.shape(deshadow_img_c_sc))
		print(tf.shape(mask_pred))'''
		#figs = [tf.reshape(tmp, (1, 256, 256, 3)), tf.reshape(deshadow_img_c_sc, (1, 256, 256, 3)), tf.reshape(mask_pred, (1, 256, 256, 3))*2, tf.reshape(gt_sc, (1, 256, 256, 3)), tf.reshape(curr_mask*255.0, (1, 256, 256, 3)), tf.reshape(curr_mask_no_hair*255.0, (1, 256, 256, 3)), tf.reshape(tf.cast(curr_mask-curr_mask_no_hair, tf.float32)*255.0, (1, 256, 256, 3)), tf.reshape(hair_region*255.0, (1, 256, 256, 3)), tf.reshape(detected_shadow_mask*255.0, (1, 256, 256, 3))]
		print(tf.shape(tmp))
		print(tf.shape(deshadow_img_c_sc))
		print(tf.shape(mask_pred))
		print(tf.shape(gt_sc))
		print(tf.shape(detected_shadow_mask))
		#mask_pred = tf.concat([mask_pred, mask_pred, mask_pred], 2)
		figs = [tf.reshape(tmp, (1, 256, 256, 3)), tf.reshape(deshadow_img_c_sc, (1, 256, 256, 3)), tf.reshape(mask_pred, (1, 256, 256, 3))*2, tf.reshape(gt_sc, (1, 256, 256, 3)), tf.reshape(detected_shadow_mask, (1, 256, 256, 3)), tf.reshape(deshadow_img_c_sc_flipped, (1, 256, 256, 3)), tf.reshape(tf.image.flip_left_right(deshadow_img_c_sc_flipped), (1, 256, 256, 3)), tf.reshape(tf.math.maximum(deshadow_img_c_sc_orig, tf.image.flip_left_right(deshadow_img_c_sc_flipped)), (1, 256, 256, 3))]
		#figs = [tf.reshape(tmp, (1, 256, 256, 3)), tf.reshape(deshadow_img_c_sc, (1, 256, 256, 3)), tf.reshape(mask_pred, (1, 256, 256, 3))*2]
		#figs = [tf.reshape(img[0,...], (1, 256, 256, 3)), tf.reshape(deshadow_img_c_sc, (1, 256, 256, 3)), tf.reshape(mask_pred, (1, 256, 256, 3))*2]
		return losses, figs, frac_nose_in_shadow, mean_intensity

	def testsfw(self, dataset_val):
		# find restore
		last_checkpoint = tf.train.latest_checkpoint(self.config.CHECKPOINT_DIR)
		if last_checkpoint:
			last_epoch = int(last_checkpoint.split('-')[-1])
			self.checkpoint.restore(last_checkpoint).expect_partial()
		else:
			last_epoch = 0
		print('**********************************************************')
		print('Restore from Epoch '+str(last_epoch))
		print('**********************************************************')

		start = time.time()
		num_list = len(dataset_val.name_list)
		for step, img_name in enumerate(dataset_val.name_list): #range(num_list):
			img, box, _ = next(dataset_val.feed)
			losses, figs = self.test_step_sfw(img, box, training=False)
			self.log.display(losses, 0, step, False, num_list)
			self.log.save_img(figs, img_name)

		print ('\n*****Time for epoch {} is {} sec*****'.format(1, int(time.time()-start)))

	def testsfw_video(self, dataset_val):
		# find restore
		last_checkpoint = tf.train.latest_checkpoint(self.config.CHECKPOINT_DIR)
		if last_checkpoint:
			last_epoch = int(last_checkpoint.split('-')[-1])
			self.checkpoint.restore(last_checkpoint).expect_partial()
		else:
			last_epoch = 0
		print('**********************************************************')
		print('Restore from Epoch '+str(last_epoch))
		print('**********************************************************')

		start = time.time()
		num_list = len(dataset_val.name_list)
		for step, img_name in enumerate(dataset_val.name_list): #range(num_list):
			img, box, _ = next(dataset_val.feed)
			losses, figs = self.test_step_sfw_video(img, box, training=False)
			self.log.display(losses, 0, step, False, num_list)
			self.log.save_img(figs, img_name)
			output_mat = {}
			output_mat['bbox'] = box.numpy()
			img_name_parts = img_name.split('/')
			scipy.io.savemat('bounding_boxes/'+img_name_parts[-2]+'_'+img_name_parts[-1]+'.mat', output_mat)

		print ('\n*****Time for epoch {} is {} sec*****'.format(1, int(time.time()-start)))

	#@tf.function
	def test_step_sfw(self, img, box, training):
		losses = {}
		figs = []
		img = tf.reshape(img, [2, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
		box = tf.reshape(box, [4])
		size = box[3] - box[1]

		img, cmap, mask, uv, reg, face = tf.split(img, [3, 3, 1, 3, 6, 1], 3)
		deshadow_img_gs, deshadow_img_c, _, mask_pred = self.gen(img, uv, reg, frame=2, share=tf.constant(True), chuck=1, training=training)
		mask_pred = mask_pred * face
		deshadow_img_c = tf.clip_by_value(deshadow_img_c, 0, 1)

		#img = img[0,...]
		#cmap = cmap[0,...]
		#deshadow_img_c = deshadow_img_c[0,...]
		masksc = mask[0,...]
		mask_predsc = mask_pred[0,...]

		ssim = tf.reduce_sum(tf.image.ssim(masksc, mask_predsc, max_val=1.0))
		psnr = tf.reduce_sum(tf.image.psnr(masksc, mask_predsc, max_val=1.0))

		masksc = tf.cast(tf.equal(masksc, 2), tf.float32)
		label = masksc
		masksc = masksc.numpy().reshape((-1))
		maskpredsc = mask_predsc.numpy().reshape((-1))
		extrsc = np.array([1,0])
		masksc = np.concatenate([extrsc,masksc],axis=0)
		maskpredsc = np.concatenate([extrsc,maskpredsc],axis=0)
		#mask = tf.equal(mask, 2)
		#fpr, tpr, thresholds = metrics.roc_curve(mask.numpy().reshape((-1)), mask_pred.numpy().reshape((-1)))
		#print(tpr)
		#input()
		#auc = metrics.auc(fpr, tpr)
		auc = metrics.roc_auc_score(masksc, maskpredsc)
		
		losses['ssim'] = ssim
		losses['psnr'] = psnr
		losses['auc']  = tf.constant(auc, tf.float32)
		figs = [img, deshadow_img_c, mask_pred*2, tf.reshape(label, (1, 256, 256, 1))]
		return losses, figs

        #@tf.function
	def test_step_sfw_video(self, img, box, training):
		losses = {}
		figs = []
		img = tf.reshape(img, [10, self.config.IMG_SIZE, self.config.IMG_SIZE, -1])
		box = tf.reshape(box, [4])
		size = box[3] - box[1]

		img, uv, reg, face = tf.split(img, [3, 3, 6, 1], 3)
		deshadow_img_gs, deshadow_img_c, _, mask_pred = self.gen(img, uv, reg, frame=10, share=tf.constant(True), chuck=1, training=training)
		mask_pred = mask_pred * face
		deshadow_img_c = tf.clip_by_value(deshadow_img_c, 0, 1)

		#img = img[0,...]
		#cmap = cmap[0,...]
		#deshadow_img_c = deshadow_img_c[0,...]
		#masksc = mask[0,...]
		mask_predsc = mask_pred[0,...]

		#ssim = tf.reduce_sum(tf.image.ssim(masksc, mask_predsc, max_val=1.0))
		#psnr = tf.reduce_sum(tf.image.psnr(masksc, mask_predsc, max_val=1.0))

		#masksc = tf.cast(tf.equal(masksc, 2), tf.float32)
		#label = masksc
		#masksc = masksc.numpy().reshape((-1))
		maskpredsc = mask_predsc.numpy().reshape((-1))
		extrsc = np.array([1,0])
		#masksc = np.concatenate([extrsc,masksc],axis=0)
		maskpredsc = np.concatenate([extrsc,maskpredsc],axis=0)
		#mask = tf.equal(mask, 2)
		#fpr, tpr, thresholds = metrics.roc_curve(mask.numpy().reshape((-1)), mask_pred.numpy().reshape((-1)))
		#print(tpr)
		#input()
		#auc = metrics.auc(fpr, tpr)
		#auc = metrics.roc_auc_score(masksc, maskpredsc)
		
		#losses['ssim'] = ssim
		#losses['psnr'] = psnr
		#losses['auc']  = tf.constant(auc, tf.float32)
		figs = [img, deshadow_img_c, mask_pred*2]
		return losses, figs

def main():
	# Base Configuration Class
	config=Config(0)
	config.CHECKPOINT_DIR = './log/FSR-OG-perlin-mask-OG-loss-weights-l1-recon-fix-BGR-to-RGB-fix-VGG-greater-augmentation-reweight-gradients-with-TSM'
	config.compile()
	# Get images and labels.
	#dataset_train = Dataset(config, 'train')
	#dataset_val   = Dataset(config, 'val')
	#dataset_test  = Dataset(config, 'test')
	dataset_test  = Dataset(config, 'test', dset='sfw')

	# model define
	fsr = FSRNet(config)
	#fsr.train(dataset_train, dataset_val)
	#fsr.test(dataset_test)
	fsr.testsfw(dataset_test)
	#fsr.testsfw_video(dataset_test)


if __name__ == "__main__":
    main()

