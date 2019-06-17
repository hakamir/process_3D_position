#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import RTSADS.Nets
import os
import sys
import time
import cv2
import json
import datetime
import shutil
from RTSADS.Data_utils import data_reader,weights_utils,preprocessing
from RTSADS.Losses import loss_factory
from RTSADS.Sampler import sampler_factory

import rospy
import message_filters
from sensor_msgs.msg import Image
from yolo_madnet.msg import DispMsg
from cv_bridge import CvBridge, CvBridgeError

import time
import sys, os
try:
    import cv2
except ImportError:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

#static params
MAX_DISP=256
PIXEL_TH = 3

class madnet:
    """
    Initialize MADNet model with tensorflow and ROS.
    """
    def __init__(self):
        # Set network parameters
        parser=argparse.ArgumentParser(description='Script for online Adaptation of a Deep Stereo Network')
        parser.add_argument("-l","--list", help='path to the list file with frames to be processed', default="lists/indoor_02/list.csv")
        parser.add_argument("-o","--output", help="path to the output folder where the results will be saved",default="output/RealSense/")
        parser.add_argument("--weights",help="path to the initial weights for the disparity estimation network",default="RTSADS/weights/MADNet/kitti/weights.ckpt")
        parser.add_argument("--ComputePerformance",help="Use the ground truth to compute the network performance",default=False)
        parser.add_argument("--modelName", help="name of the stereo model to be used", default="MADNet", choices=RTSADS.Nets.STEREO_FACTORY.keys())
        parser.add_argument("--numBlocks", help="number of CNN portions to train at each iteration",type=int,default=1)
        parser.add_argument("--lr", help="value for learning rate",default=0.0001, type=float)
        parser.add_argument("--blockConfig",help="path to the block_config json file",default="RTSADS/block_config/MadNet_full.json")
        parser.add_argument("--sampleMode",help="choose the sampling heuristic to use",choices=sampler_factory.AVAILABLE_SAMPLER,default='PROBABILITY')
        parser.add_argument("--fixedID",help="index of the portions of network to train, used only if sampleMode=FIXED",type=int,nargs='+',default=[0])
        parser.add_argument("--reprojectionScale",help="compute all loss function at 1/reprojectionScale",default=1,type=int)
        parser.add_argument("--summary",help='flag to enable tensorboard summaries',action='store_true')
        parser.add_argument("--imageShape", help='two int for image shape [height,width]', nargs='+', type=int, default=[360,640])
        parser.add_argument("--SSIMTh",help="reset network to initial configuration if loss is above this value",type=float,default=0.5)
        parser.add_argument("--sampleFrequency",help="sample new network portions to train every K frame",type=int,default=1)
        parser.add_argument("--mode",help="online adaptation mode: NONE - perform only inference, FULL - full online backprop, MAD - backprop only on portions of the network", choices=['NONE','FULL','MAD'], default='MAD')
        parser.add_argument("--logDispStep", help="save disparity every K step, -1 to disable", default=1, type=int)
        self.args=parser.parse_args()

        # Load MADNet
        self.load_model(self.args)

        self.bridge = CvBridge()
        rospy.init_node('madnet_node')
        sub_l = message_filters.Subscriber('/d435/infra1/image_rect_raw', Image)
        sub_r = message_filters.Subscriber('/d435/infra2/image_rect_raw', Image)
        ats = message_filters.ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=1, slop=0.01)
        ats.registerCallback(self.process)
        self.pub = rospy.Publisher('/disparity', Image, queue_size=1)
        self.msg_pub = Image()
        rospy.spin()

    def scale_tensor(self,tensor,scale):
	    return preprocessing.rescale_image(tensor,[tf.shape(tensor)[1]//scale,tf.shape(tensor)[2]//scale])

    def softmax(self, x):
    	"""Compute softmax values for each sets of scores in x."""
    	return np.exp(x) / np.sum(np.exp(x), axis=0)

    def load_model(self, args):
    	#load json file config
    	with open(args.blockConfig) as json_data:
    		train_config = json.load(json_data)

    	#read input data
    	with tf.variable_scope('input_reader'):

    		if args.ComputePerformance:
    			self.left_img_batch, self.right_img_batch, self.gt_image_batch = data_set.get_batch()
    			inputs={
    				'left':self.left_img_batch,
    				'right':self.right_img_batch,
    				'target':self.gt_image_batch
    				}
    		else:
    			#left_img_batch, right_img_batch, _ = data_set.get_batch()
    			self.height,self.width=args.imageShape
    			self.left_img_batch = tf.placeholder(tf.float32,shape=[1,self.height,self.width,3], name='left_input')
    			self.right_img_batch = tf.placeholder(tf.float32, shape=[1,self.height,self.width,3], name='right_input')
    			inputs={
    			'left':self.left_img_batch,
    			'right':self.right_img_batch}

    	#build inference network
    	with tf.variable_scope('model'):
    		net_args = {}
    		net_args['left_img'] = self.left_img_batch
    		net_args['right_img'] = self.right_img_batch
    		net_args['split_layers'] = [None]
    		net_args['sequence'] = True
    		net_args['train_portion'] = 'BEGIN'
    		net_args['bulkhead'] = True if args.mode=='MAD' else False
    		stereo_net = RTSADS.Nets.get_stereo_net(args.modelName, net_args)
    		print('Stereo Prediction Model:\n', stereo_net)
    		predictions = stereo_net.get_disparities()
    		self.full_res_disp = predictions[-1]

    	#build real full resolution loss
    	with tf.variable_scope('full_res_loss'):
    		# reconstruction loss between warped right image and original left image
    		self.full_reconstruction_loss =  loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)(predictions,inputs)


    	#build validation ops
    	if args.ComputePerformance:
    		with tf.variable_scope('validation_error'):
    			# compute error against gt
    			abs_err = tf.abs(self.full_res_disp - gt_image_batch)
    			valid_map = tf.where(tf.equal(gt_image_batch, 0), tf.zeros_like(gt_image_batch, dtype=tf.float32), tf.ones_like(gt_image_batch, dtype=tf.float32))
    			filtered_error = abs_err * valid_map

    			abs_err = tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
    			bad_pixel_abs = tf.where(tf.greater(filtered_error, PIXEL_TH), tf.ones_like(filtered_error, dtype=tf.float32), tf.zeros_like(filtered_error, dtype=tf.float32))
    			bad_pixel_perc = tf.reduce_sum(bad_pixel_abs) / tf.reduce_sum(valid_map)

    	#build train ops
    	disparity_trainer = tf.train.MomentumOptimizer(args.lr,0.9)
    	self.train_ops = []
    	if args.mode == 'MAD':
    		#build train ops for separate portion of the network
    		predictions = predictions[:-1] #remove full res disp
    		if args.ComputePerformance:
    			inputs_modules = {
    				'left':self.scale_tensor(self.left_img_batch,args.reprojectionScale),
    				'right':self.scale_tensor(self.right_img_batch,args.reprojectionScale),
    				'target':self.scale_tensor(self.gt_image_batch,args.reprojectionScale)/args.reprojectionScale
    			}
    		else:
    			inputs_modules = {
    				'left':self.scale_tensor(self.left_img_batch,args.reprojectionScale),
    				'right':self.scale_tensor(self.right_img_batch,args.reprojectionScale),
    			}
    		assert(len(predictions)==len(train_config))
    		for counter,p in enumerate(predictions):
    			print('Build train ops for disparity {}'.format(counter))

    			#rescale predictions to proper resolution
    			multiplier = tf.cast(tf.shape(self.left_img_batch)[1]//tf.shape(p)[1],tf.float32)
    			p = preprocessing.resize_to_prediction(p,inputs_modules['left'])*multiplier

    			#compute reprojection error
    			with tf.variable_scope('reprojection_'+str(counter)):
    				reconstruction_loss = loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)([p],inputs_modules)

    			#build train op
    			layer_to_train = train_config[counter]
    			print('Going to train on {}'.format(layer_to_train))
    			var_accumulator=[]
    			for name in layer_to_train:
    				var_accumulator+=stereo_net.get_variables(name)
    			print('Number of variable to train: {}'.format(len(var_accumulator)))

    			#add new training op
    			self.train_ops.append(disparity_trainer.minimize(reconstruction_loss,var_list=var_accumulator))

    			print('Done')
    			print('='*50)

    		#create Sampler to fetch portions to train
    		self.sampler = sampler_factory.get_sampler(args.sampleMode,args.numBlocks,args.fixedID)

    	elif args.mode=='FULL':
    		#build single train op for the full network
    		self.train_ops.append(disparity_trainer.minimize(self.full_reconstruction_loss))


    	if args.summary:
    		tf.summary.image('self.full_res_disp',preprocessing.colorize_img(self.full_res_disp,cmap='jet'),max_outputs=1)
    		if args.ComputePerformance:
    			#add summaries
    			tf.summary.scalar('EPE',abs_err)
    			tf.summary.scalar('bad3',bad_pixel_perc)
    			tf.summary.image('gt_disp',preprocessing.colorize_img(gt_image_batch,cmap='jet'),max_outputs=1)

    		#create summary logger
    		summary_op = tf.summary.merge_all()
    		logger = tf.summary.FileWriter(args.output)


    	#start session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #init stuff
        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        #restore disparity inference weights
        var_to_restore = weights_utils.get_var_to_restore_list(args.weights, [])
        assert(len(var_to_restore)>0)
        restorer = tf.train.Saver(var_list=var_to_restore)
        restorer.restore(self.sess,args.weights)
        print('Disparity Net Restored?: {}, number of restored variables: {}'.format(True,len(var_to_restore)))

        num_actions=len(self.train_ops)
        if args.mode=='FULL':
        	selected_train_ops = self.train_ops
        else:
        	selected_train_ops = [tf.no_op()]

        self.epe_accumulator = []
        self.bad3_accumulator = []
        self.time_accumulator = []
        self.exec_time = 0
        self.fetch_counter=[0]*num_actions
        self.sample_distribution=np.zeros(shape=[num_actions])
        temp_score = np.zeros(shape=[num_actions])
        self.loss_t_2 = 0
        self.loss_t_1 = 0
        self.expected_loss = 0
        self.last_trained_blocks = []
        self.reset_counter=0
        self.step=0

    """
    ROS process. This function perform distance estimation with MADNet working
    through Tensorflow.

    See the GitHub of the author:
    https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo
    """

    def process(self, msg_l, msg_r):
        try:
            # Input image from topic
            start = time.time()
            img_l = self.bridge.imgmsg_to_cv2(msg_l, "bgr8")
            img_r = self.bridge.imgmsg_to_cv2(msg_r, "bgr8")
            #fetch new network portion to train
            if self.step%self.args.sampleFrequency==0 and self.args.mode=='MAD':
                #Sample
                distribution = self.softmax(self.sample_distribution)
                blocks_to_train = self.sampler.sample(distribution)
                selected_train_ops = [self.train_ops[i] for i in blocks_to_train]

                #accumulate sampling statistics
                for l in blocks_to_train:
                    self.fetch_counter[l]+=1

            #build list of tensorflow operations that needs to be executed

            if self.args.ComputePerformance:
                #errors and full resolution loss
                tf_fetches = [abs_err,bad_pixel_perc,self.full_reconstruction_loss]

            else:
                tf_fetches = [self.full_reconstruction_loss]

            if self.args.summary and self.step%100==0:
                #summaries
                tf_fetches = tf_fetches + [summary_op]

            #update ops
            tf_fetches = tf_fetches+selected_train_ops

            if self.args.logDispStep!=-1 and self.step%self.args.logDispStep==0:
                #prediction for serialization to disk
                tf_fetches=tf_fetches + [self.full_res_disp]

            left_image=np.zeros((1,self.height,self.width,3))
            left_image[0,:,:,:] = img_l
            right_image=np.zeros((1,self.height,self.width,3))
            right_image[0,:,:,:] = img_r
            fd = {
                self.left_img_batch: left_image,
                self.right_img_batch: right_image
            }

            #run network
            fetches = self.sess.run(tf_fetches, feed_dict=fd)
            if self.args.ComputePerformance:
                new_loss = fetches[2]
            else:
                new_loss = fetches[0]

            if self.args.mode == 'MAD':
                #update sampling probabilities
                if self.step==0:
                    self.loss_t_2 = new_loss
                    self.loss_t_1 = new_loss
                self.expected_loss = 2*self.loss_t_1-self.loss_t_2
                gain_loss=self.expected_loss-new_loss
                self.sample_distribution = 0.99*self.sample_distribution
                for i in self.last_trained_blocks:
                    self.sample_distribution[i] += 0.01*gain_loss

                self.last_trained_blocks=blocks_to_train
                self.loss_t_2 = self.loss_t_1
                self.loss_t_1 = new_loss


            if self.args.ComputePerformance:
                #accumulate performance metrics
                self.epe_accumulator.append(fetches[0])
                self.bad3_accumulator.append(fetches[1])

            if self.step%100==0:
                #log on terminal
                if self.args.summary:
                    if self.args.ComputePerformance:
                        logger.add_summary(fetches[3],global_step=self.step)
                    else :
                        logger.add_summary(fetches[1],global_step=self.step)

            #reset network if necessary
            if new_loss>self.args.SSIMTh:
                restorer.restore(self.sess,self.args.weights)
                self.reset_counter+=1

            #save disparity if requested
            if self.args.logDispStep!=-1 and self.step%self.args.logDispStep==0:
                dispy=fetches[-1]
                dispy_to_save = np.clip(dispy[0].astype(np.uint16), 0, MAX_DISP)
                out_img=dispy_to_save*255
                print(out_img)
                self.pub.publish(self.bridge.cv2_to_imgmsg(out_img, "16UC1"))
            self.step+=1
            print('FPS: {}'.format(int(1 / (time.time() - start))))
        except tf.errors.OutOfRangeError:
        	pass
    	finally:
        	self.epe_array = self.epe_accumulator
        	self.bad3_array = self.bad3_accumulator
        	self.epe_accumulator = np.sum(self.epe_accumulator)
        	self.bad3_accumulator = np.sum(self.bad3_accumulator)


if __name__ == '__main__':
    try:
        madnet()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start distance estimation node.')
