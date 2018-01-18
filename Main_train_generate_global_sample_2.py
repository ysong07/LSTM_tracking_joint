import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
from data_handler_inference_good_only import *
from params_refine_local_2 import * 
from G_model_glob_share_weight import *
from G_inference_new2 import *
#from G_model_local_train import *
import os 
import time
import sys

def get_center(mask):
  yv,xv = np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]))
  xv = np.float32(xv)
  yv = np.float32(yv)

  center_x,center_y = np.int32(np.sum(xv*mask)/mask.sum()),np.int32(np.sum(yv*mask)/mask.sum())
  center = [center_y,center_x]
  return center


def crop_image_include_coundary(output,input,center):
  height,width = output.shape[0],output.shape[1]
  min_x = max(0,int(center[0]-height/2))
  min_out_x = max(0,int(height/2-center[0]))

  min_y = max(0,int(center[1]-width/2))
  min_out_y = max(0,int(width/2-center[1]))

  max_x = min(input.shape[0],int(center[0]+height/2))
  max_out_x = min(height, height+ input.shape[0]- int(center[0]+height/2))
  max_y =  min(input.shape[1],int(center[1]+width/2))
  max_out_y = min(width, width+ input.shape[1]- int(center[1]+width/2))

  try:
    output[min_out_x:max_out_x,min_out_y:max_out_y,:] = input[min_x:max_x,min_y:max_y,:]
  except:
    pdb.set_trace()

  return output

class Inference:
  def __init__(self,train_file_name):
      self.sess = tf.Session()

      kernel_size_dec = []
      kernel_size_dec.append([5,5])
      kernel_size_dec.append([5,5])
      kernel_size_dec.append([5,5])
      kernel_size_dec.append([5,5])
      self.h_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
      self.c_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
      self.h_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
      self.c_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
      """ placeholder for feature extractor """
      self.batch_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,3])
      self.mask_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,64])
      self.output_feature_1 = feature_extract(self.batch_frames,self.mask_frames)

      self.no_mask_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,64])
      self.output_feature_0 = feature_extract(self.batch_frames,self.no_mask_frames)
      input_feature = feature_extract(self.batch_frames,self.no_mask_frames)

      with tf.variable_scope('trainable_params_local') as scope:

        output_feature = tf.concat([self.output_feature_0,self.output_feature_1],axis=3)
        with tf.variable_scope('initial_0'):

            c_matrix_0_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_0_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
            c_0_0 = tf.nn.conv2d(output_feature,c_matrix_0_0,strides=[1,1,1,1], padding='SAME') + c_bias_0_0

            c_0_0 = tf.nn.relu(c_0_0)
            c_matrix_0_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_0_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.c_0 = tf.tanh(tf.nn.conv2d(c_0_0,c_matrix_0_1,strides=[1,1,1,1], padding='SAME') + c_bias_0_1)

            h_matrix_0_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_0_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
            h_0_0 = tf.nn.relu(tf.nn.conv2d(output_feature,h_matrix_0_0,strides=[1,1,1,1], padding='SAME') + h_bias_0_0)

            h_matrix_0_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_0_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.h_0 = tf.tanh(tf.nn.conv2d(h_0_0, h_matrix_0_1,strides=[1,1,1,1], padding='SAME') + h_bias_0_1)

        with tf.variable_scope('initial_1'):
            c_matrix_1_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_1_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
            c_1_0 = tf.nn.conv2d(output_feature,c_matrix_1_0,strides=[1,1,1,1], padding='SAME') + c_bias_1_0

            c_1_0 = tf.nn.relu(c_1_0)
            c_matrix_1_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_1_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.c_1 = tf.tanh(tf.nn.conv2d(c_1_0,c_matrix_1_1,strides=[1,1,1,1], padding='SAME') + c_bias_1_1)

            h_matrix_1_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_1_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
            h_1_0 = tf.nn.relu(tf.nn.conv2d(output_feature,h_matrix_1_0,strides=[1,1,1,1], padding='SAME') + h_bias_1_0)

            h_matrix_1_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_1_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.h_1 = tf.tanh(tf.nn.conv2d(h_1_0, h_matrix_1_1,strides=[1,1,1,1], padding='SAME') + h_bias_1_1)


        self.G_model = G_model_(scope="G_model",height=14,width=14,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[256,256],kernel_size_dec=kernel_size_dec,num_dec_input=[256,128,64,32],num_dec_output=[128,64,32,1],layer_num_cnn =4,initial_h_0=self.h_0,initial_c_0=self.c_0,initial_h_1=self.h_1,initial_c_1=self.c_1,img_height=200,img_width=200)

      variable_collection_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_local")
      
      saver_local = tf.train.Saver(variable_collection_local)
      #saver_local.restore(self.sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_1/test_refine_total_0/model.ckpt-40000")
      saver_local.restore(self.sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_refine_total_large2_test/model_keep.ckpt-24")	
      self.glob_h_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
      self.glob_c_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
      self.glob_h_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
      self.glob_c_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
      
      with tf.variable_scope('trainable_params_glob') as scope:
        self.first_feature = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,1])
        self.rest_feature = tf.placeholder(tf.float32,shape=[FLAGS.batch_size*1,224,224,1])
        with tf.variable_scope('feature_extract') as f_scope:
          def feature_extract_mask(batch_frames):
            def avg_pool( bottom, name):
              return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
      
            def max_pool( bottom, name):
              return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
      
            def conv_layer(bottom, name,shape):
              with tf.variable_scope(name):
                filt = get_conv_filter(name,shape)
      
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
                conv_biases = get_bias(name,shape)
                bias = tf.nn.bias_add(conv, conv_biases)
                relu = tf.nn.relu(bias)
                return relu
            def get_conv_filter(name,shape):
              return tf.get_variable(name+"_matrix",shape = shape,initializer = tf.random_uniform_initializer(-0.01, 0.01))
      
            def get_bias(name,shape):
              return tf.get_variable(name+"_bias",shape = shape[3],initializer=tf.constant_initializer(0.01))
                        #return tf.constant(data_dict[name][1], name="biases")
      
            conv1_1 = conv_layer(batch_frames, "conv1_1",[3,3,1,8])
            conv1_2 = conv_layer(conv1_1, "conv1_2",[3,3,8,8])
            pool1 = max_pool(conv1_2, 'pool1')
      
            conv2_1 = conv_layer(pool1, "conv2_1",[3,3,8,16])
            conv2_2 = conv_layer(conv2_1, "conv2_2",[3,3,16,16])
            pool2 = max_pool(conv2_2, 'pool2')
      
            conv3_1 = conv_layer(pool2, "conv3_1",[3,3,16,32])
            conv3_2 = conv_layer(conv3_1, "conv3_2",[3,3,32,32])
            pool3 = max_pool(conv3_2, 'pool3')
      
            conv4_1 = conv_layer(pool3, "conv4_1",[3,3,32,64])
            conv4_2 = conv_layer(conv4_1, "conv4_2",[3,3,64,64])
            pool4 = max_pool(conv4_2, 'pool4')
      
            return pool4
          glob_feature_0 = feature_extract_mask(self.first_feature)
          f_scope.reuse_variables()
          frame_features = feature_extract_mask(self.rest_feature)
        with tf.variable_scope('initial_0'):
          c_matrix_0 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_0 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_c_0 = tf.nn.conv2d(glob_feature_0,c_matrix_0,strides=[1,1,1,1], padding='SAME') + c_bias_0
      
          h_matrix_0 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_0 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_h_0 = tf.tanh(tf.nn.conv2d(glob_feature_0,h_matrix_0,strides=[1,1,1,1], padding='SAME') + h_bias_0)
      
        with tf.variable_scope('initial_1'):
          c_matrix_1 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_1 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_c_1 = tf.nn.conv2d(glob_feature_0,c_matrix_1,strides=[1,1,1,1], padding='SAME') + c_bias_1
      
          h_matrix_1 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_1 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_h_1 = tf.tanh(tf.nn.conv2d(glob_feature_0,h_matrix_1,strides=[1,1,1,1], padding='SAME') + h_bias_1)
      
        self.G_model_G = G_model_glob(scope="G_model",height=14,width=14,length=1,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[64,64],initial_h_0=self.glob_h_0,initial_c_0=self.glob_c_0,initial_h_1=self.glob_h_1,initial_c_1=self.glob_c_1,input_features=frame_features)
      
      
      variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_glob")
      """saver """
      saver = tf.train.Saver(variable_collection)
      saver.restore(self.sess,"/scratch/ys1297/LSTM_tracking/source_cross_resnet_globe/checkpoints/iter_2/test_new_no_smooth_share_feature_weight_2/model.ckpt-27000")
      sess1 = None
      self.data_handler = data_handler_(sess1,batch_size=FLAGS.batch_size,length = 2,train_file_name = train_file_name) # get batch data

  def define_graph(self):
 
      sess1 = None

      image, mask = self.data_handler.Get_all()
      W = image.shape[1]
      H = image.shape[2]
	
	
      kernel_size_dec = []
      kernel_size_dec.append([5,5])
      kernel_size_dec.append([5,5])
      kernel_size_dec.append([5,5])
      kernel_size_dec.append([5,5])
      self.h_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])
      self.c_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])
      self.h_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])
      self.c_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])
      """ placeholder for feature extractor """
      self.batch_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,3])
      self.mask_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,64])
      self.output_feature_1 = feature_extract(self.batch_frames,self.mask_frames)

      self.no_mask_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,64])
      self.output_feature_0 = feature_extract(self.batch_frames,self.no_mask_frames)
      with tf.variable_scope('trainable_params_local') as scope:

        output_feature = tf.concat([self.output_feature_0,self.output_feature_1],axis=3)
        scope.reuse_variables()
        with tf.variable_scope('initial_0'):

            c_matrix_0_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_0_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
            c_0_0 = tf.nn.conv2d(output_feature,c_matrix_0_0,strides=[1,1,1,1], padding='SAME') + c_bias_0_0

            c_0_0 = tf.nn.relu(c_0_0)
            c_matrix_0_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_0_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.c_0 = tf.tanh(tf.nn.conv2d(c_0_0,c_matrix_0_1,strides=[1,1,1,1], padding='SAME') + c_bias_0_1)

            h_matrix_0_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_0_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
            h_0_0 = tf.nn.relu(tf.nn.conv2d(output_feature,h_matrix_0_0,strides=[1,1,1,1], padding='SAME') + h_bias_0_0)

            h_matrix_0_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_0_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.h_0 = tf.tanh(tf.nn.conv2d(h_0_0, h_matrix_0_1,strides=[1,1,1,1], padding='SAME') + h_bias_0_1)

        with tf.variable_scope('initial_1'):
            c_matrix_1_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_1_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
            c_1_0 = tf.nn.conv2d(output_feature,c_matrix_1_0,strides=[1,1,1,1], padding='SAME') + c_bias_1_0

            c_1_0 = tf.nn.relu(c_1_0)
            c_matrix_1_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_1_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.c_1 = tf.tanh(tf.nn.conv2d(c_1_0,c_matrix_1_1,strides=[1,1,1,1], padding='SAME') + c_bias_1_1)

            h_matrix_1_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_1_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
            h_1_0 = tf.nn.relu(tf.nn.conv2d(output_feature,h_matrix_1_0,strides=[1,1,1,1], padding='SAME') + h_bias_1_0)

            h_matrix_1_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_1_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
            self.h_1 = tf.tanh(tf.nn.conv2d(h_1_0, h_matrix_1_1,strides=[1,1,1,1], padding='SAME') + h_bias_1_1)


        self.G_model = G_model_(scope="G_model",height=14,width=14,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[256,256],kernel_size_dec=kernel_size_dec,num_dec_input=[256,128,64,32],num_dec_output=[128,64,32,1],layer_num_cnn =4,initial_h_0=self.h_0,initial_c_0=self.c_0,initial_h_1=self.h_1,initial_c_1=self.c_1,img_height=W,img_width=H)
 
      self.glob_h_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
      self.glob_c_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
      self.glob_h_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
      self.glob_c_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])

      with tf.variable_scope('trainable_params_glob') as scope:
	scope.reuse_variables()
        self.first_feature = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,1])
        self.rest_feature = tf.placeholder(tf.float32,shape=[FLAGS.batch_size*(1),224,224,1])
        with tf.variable_scope('feature_extract') as f_scope:
          def feature_extract_mask(batch_frames):
            def avg_pool( bottom, name):
              return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

            def max_pool( bottom, name):
              return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

            def conv_layer(bottom, name,shape):
              with tf.variable_scope(name):
                filt = get_conv_filter(name,shape)

                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
                conv_biases = get_bias(name,shape)
                bias = tf.nn.bias_add(conv, conv_biases)
                relu = tf.nn.relu(bias)
                return relu
            def get_conv_filter(name,shape):
              return tf.get_variable(name+"_matrix",shape = shape,initializer = tf.random_uniform_initializer(-0.01, 0.01))

            def get_bias(name,shape):
              return tf.get_variable(name+"_bias",shape = shape[3],initializer=tf.constant_initializer(0.01))
                    #return tf.constant(data_dict[name][1], name="biases")

            conv1_1 = conv_layer(batch_frames, "conv1_1",[3,3,1,8])
            conv1_2 = conv_layer(conv1_1, "conv1_2",[3,3,8,8])
            pool1 = max_pool(conv1_2, 'pool1')

            conv2_1 = conv_layer(pool1, "conv2_1",[3,3,8,16])
            conv2_2 = conv_layer(conv2_1, "conv2_2",[3,3,16,16])
            pool2 = max_pool(conv2_2, 'pool2')

            conv3_1 = conv_layer(pool2, "conv3_1",[3,3,16,32])
            conv3_2 = conv_layer(conv3_1, "conv3_2",[3,3,32,32])
            pool3 = max_pool(conv3_2, 'pool3')

            conv4_1 = conv_layer(pool3, "conv4_1",[3,3,32,64])
            conv4_2 = conv_layer(conv4_1, "conv4_2",[3,3,64,64])
            pool4 = max_pool(conv4_2, 'pool4')

            return pool4
          f_scope.reuse_variables()
	  glob_feature_0 = feature_extract_mask(self.first_feature)
          frame_features = feature_extract_mask(self.rest_feature)
	scope.reuse_variables()
        with tf.variable_scope('initial_0'):
          c_matrix_0 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_0 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_c_0 = tf.nn.conv2d(glob_feature_0,c_matrix_0,strides=[1,1,1,1], padding='SAME') + c_bias_0

          h_matrix_0 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_0 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_h_0 = tf.tanh(tf.nn.conv2d(glob_feature_0,h_matrix_0,strides=[1,1,1,1], padding='SAME') + h_bias_0)

        with tf.variable_scope('initial_1'):
          c_matrix_1 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_1 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_c_1 = tf.nn.conv2d(glob_feature_0,c_matrix_1,strides=[1,1,1,1], padding='SAME') + c_bias_1

          h_matrix_1 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_1 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
          self.glob_h_1 = tf.tanh(tf.nn.conv2d(glob_feature_0,h_matrix_1,strides=[1,1,1,1], padding='SAME') + h_bias_1)

        self.G_model_G = G_model_glob(scope="G_model",height=14,width=14,length=1,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[64,64],initial_h_0=self.glob_h_0,initial_c_0=self.glob_c_0,initial_h_1=self.glob_h_1,initial_c_1=self.glob_c_1,input_features=frame_features)

  def provide_data(self):  
#      variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_glob")
#      """saver """
#      saver = tf.train.Saver(variable_collection)

      image, mask = self.data_handler.Get_all()
      image_template_list = []
      mask_template_list = []
      feature_template_list = []

      image_true_list = []
      mask_true_list =[]
      mask_predict_list =[]	
      for step in xrange(32):
        if step ==0:
          masks_resize = cv2.resize(mask[0,:,:,0],(224,224)).reshape(1,224,224,1)
          image_resize = cv2.resize(image[0,:,:,:],(224,224)).reshape(1,224,224,3)
          
          ratio = np.array([float(mask.shape[1])/mask.shape[2]]).reshape(1,1)

	  # append ground truth first frame
          mask_true_list.append(masks_resize)
          image_true_list.append(image_resize) 
 
	  G_glob_dict = {self.rest_feature:masks_resize, self.G_model_G.ratio:ratio,self.first_feature:masks_resize}
          glob_frame,glob_cell,glob_crop_size,glob_center = self.sess.run([self.G_model_G.logits,self.G_model_G.cell_state,self.G_model_G.crop_size,self.G_model_G.center],feed_dict= G_glob_dict)
          
          #temp=cv2.resize(glob_frame.squeeze(),(image.shape[1],image.shape[2]))
          #np.where(temp.sum(      
    
          crop_size = int(image.shape[1]/glob_crop_size/224.0*64)*2
          #crop_size = np.asarray(glob_crop_size*64*mask.shape[1]/224)
          center = np.zeros([2],dtype = 'int32')
          temp = (1-glob_center[0,0]/glob_crop_size*ratio)/2.0 ## location at 224 by 224
          center[0] = temp *image.shape[2]
          
          temp = (1-glob_center[0,1]/glob_crop_size)/2.0 ## location at 224 by 224
          center[1] = temp *image.shape[1]

      
          # crop first template(ground truth)
          first_template = crop_image_include_coundary(np.zeros((crop_size,crop_size,3)),image.squeeze(),(center[1],center[0]))
          first_template = cv2.resize(first_template,(224,224)).reshape(FLAGS.batch_size,224,224,3)

          mask_temp = mask[0,:,:,0:1]>0
          first_mask = crop_image_include_coundary(np.zeros((crop_size,crop_size,1)),mask_temp,(center[1],center[0]))
          first_mask = cv2.resize(first_mask,(224,224)).reshape(224,224,1)
           
          
          first_mask = np.tile(first_mask,(1,1,64)).reshape(FLAGS.batch_size,224,224,64)	
          	
          feed_dict = {self.batch_frames:first_template,self.mask_frames:first_mask}
          feature_mask = self.sess.run(self.output_feature_1,feed_dict=feed_dict)
    
          feed_dict = {self.batch_frames:first_template,self.no_mask_frames:np.ones((1,224,224,64))}
          feature_no_mask = self.sess.run(self.output_feature_0,feed_dict=feed_dict)

          self.data_handler.set_id(step+1) 
          image, mask = self.data_handler.Get_all()

         
          feed_dict ={self.G_model.input_frame:image,self.output_feature_1:feature_mask,self.output_feature_0:feature_no_mask,self.G_model.center:center[::-1],self.G_model.crop_size:np.asarray(crop_size).reshape(1),self.G_model.input_mask:mask[:,:,:,0:1]}
          g_predict,g_templates,g_states,g_original_template,g_mask_template,g_feature_template = self.sess.run([self.G_model.predicts,self.G_model.templates,self.G_model.cell_states,self.G_model.original_template,self.G_model.mask_template,self.G_model.feature_list],feed_dict= feed_dict)		 

          #g_predict =temp
          image_template_list.append(g_original_template)
          mask_template_list.append(g_mask_template)
	  feature_template_list.append(g_feature_template)
        else:
          #masks_resize = cv2.resize(mask[0,:,:,0],(224,224)).reshape(1,224,224,1)
          masks_resize = np.zeros([1,224,224,1])
          temp = cv2.resize(g_predict,(224,224)).reshape(1,224,224,1)
          masks_resize[temp>=0.5]=255
          #masks_resize = cv2.resize(g_predict,(224,224)).reshape(1,224,224,1)
         
	  mask_true_list.append(cv2.resize(mask[0,:,:,0],(224,224)).reshape(1,224,224,1))
	  image_true_list.append(cv2.resize(image[0,:,:,:],(224,224)).reshape(1,224,224,3)) 
	  mask_predict_list.append(temp*255.0)

          
          self.data_handler.set_id(step+1)
          image, mask = self.data_handler.Get_all()

          ratio = np.array([float(mask.shape[1])/mask.shape[2]]).reshape(1,1)

          G_glob_dict = {self.rest_feature:masks_resize, self.G_model_G.ratio:ratio,self.glob_c_0:glob_cell[0:1,0,:,:,0:64],self.glob_h_0:glob_cell[0:1,0,:,:,64:128],self.glob_c_1:glob_cell[1:2,0,:,:,0:64],self.glob_h_1:glob_cell[1:2,0,:,:,64:128]}
          glob_frame,glob_cell,glob_crop_size,glob_center = self.sess.run([self.G_model_G.logits,self.G_model_G.cell_state,self.G_model_G.crop_size,self.G_model_G.center],feed_dict= G_glob_dict)


          old_crop_size = crop_size
          old_center = np.asarray(center)
  #        crop_size = 4*np.int(np.sqrt((g_predict.squeeze()>0.4).sum()))
#          crop_size = int(image.shape[1]/glob_crop_size/224.0*64)*2
	  crop_size = 4*np.int(np.sqrt((mask[0,:,:,0].squeeze()>0.4).sum()))
          crop_size = int((crop_size*0.0+old_crop_size*1.0))
          crop_size = min(crop_size,224*3)
          crop_size = max(crop_size,112)
  
  
  
          center = np.zeros([2],dtype = 'int32')
          temp = (1-glob_center[0,0]/glob_crop_size*ratio)/2.0 ## location at 224 by 224
          center[0] = temp *image.shape[2]
          temp = (1-glob_center[0,1]/glob_crop_size)/2.0 ## location at 224 by 224
          center[1] = temp *image.shape[1]
          center = get_center(mask[0,:,:,0].squeeze())
	  center[0] = max(40,center[0])
          center[0] = min(image.shape[2]-40,center[0])
          center[1] = max(40,center[1])
          center[1] = min(image.shape[1]-40,center[1])
	  if step<=5:
	    center = ((old_center*0.0+ np.asarray(center)*1.0)).tolist()
	  else:	
            center = ((old_center*0.1+ np.asarray(center)*0.9)).tolist()

          feed_dict ={self.c_0:g_states[0,:,:,:,:,0:256].reshape(FLAGS.batch_size,14,14,256),self.h_0:g_states[0,:,:,:,:,256:512].reshape(FLAGS.batch_size,14,14,256),self.c_1:g_states[1,:,:,:,:,0:256].reshape(FLAGS.batch_size,14,14,256),self.h_1:g_states[1,:,:,:,:,256:512].reshape(FLAGS.batch_size,14,14,256),self.G_model.center:center[::-1],self.G_model.crop_size:np.asarray(crop_size).reshape(1),self.G_model.input_frame:image,self.G_model.input_mask:mask[:,:,:,0:1]}
          g_predict,g_templates,g_states,g_original_template,g_mask_template,g_feature_template = self.sess.run([self.G_model.predicts,self.G_model.templates,self.G_model.cell_states,self.G_model.original_template,self.G_model.mask_template,self.G_model.feature_list],feed_dict= feed_dict)
          image_template_list.append(g_original_template)
          mask_template_list.append(g_mask_template)
	  
          feature_template_list.append(g_feature_template)  
          

      masks_resize = np.zeros([1,224,224,1])
      temp = cv2.resize(g_predict,(224,224)).reshape(1,224,224,1)
      masks_resize[temp>=0.5]=255
      mask_predict_list.append(temp*255.0)

      image_true_list.append(cv2.resize(image[0,:,:,:],(224,224)).reshape(1,224,224,3))
      mask_true_list.append(cv2.resize(mask[0,:,:,0],(224,224)).reshape(1,224,224,1))
      self.data_handler.set_indices_id()
    
      return np.stack(image_true_list,axis=1),np.stack(mask_true_list,axis=1) ,np.stack(mask_predict_list,axis=1),ratio	
      #return np.stack(image_template_list,axis=1),np.stack(mask_template_list,axis=0),feature_mask,feature_no_mask,np.stack(feature_template_list,axis=1)	

   
      	

def main(argv=None):  # pylint: disable=unused-argument
#      if tf.gfile.Exists(FLAGS.train_dir):
#        tf.gfile.DeleteRecursively(FLAGS.train_dir)
#      tf.gfile.MakeDirs(FLAGS.train_dir) 

    
      with tf.Graph().as_default():
	sess1 = tf.Session()


	with open(FLAGS.test_file_dir) as file:
	  file_list = file.readlines()	
	  file_name = file_list[argv][0:-1]


	  Infer =  Inference(file_name)
	  Infer.define_graph()
	  file_new = h5py.File('/scratch/ys1297/LSTM_tracking/data_global_refine/'+file_name)
	  file_new.create_dataset('image',((Infer.data_handler.indices_).size,33,224,224,3))
	  file_new.create_dataset('true_mask',((Infer.data_handler.indices_).size,33,224,224,1))
	  file_new.create_dataset('predict_mask',((Infer.data_handler.indices_).size,32,224,224,1))
	  file_new.create_dataset('valid_num',(1,),dtype='int')
          file_new.create_dataset('ratio',(1,))
 	
	  count = -1
	  np.random.seed(10)
	  for step in range(1000):
	    try:
              t1 = time.time()
              image_array,masks,predict_mask,ratio= Infer.provide_data()	
	   
	      count = count+1
              file_new['image'][count]  = image_array
              file_new['true_mask'][count] =  masks
              file_new['predict_mask'][count] =  predict_mask
	      file_new['ratio'][0] = ratio 
	      #g_summary,_,g_loss = sess1.run([summary,train_op,loss],feed_dict= G_feed_dict)
              #summary_writer.add_summary(g_summary, step)
	      #print g_loss
              #t2 =  time.time()
              #print t2-t1
            except:
	      Infer.data_handler.set_indices_id()
	      print "error indices id: " + str(Infer.data_handler.indices_id)
	      
	    if count == (Infer.data_handler.indices_).size or Infer.data_handler.indices_id>=(Infer.data_handler.indices_).size:
	      file_new['valid_num'][0] = count
	      file_new.close()
	      break
	    print count
	    #if step %0 ==0:
	    #  tf.reset_default_graph()
	    #  Infer =  Inference()
	      
	    #if step %100==0:
            #  checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            #  saver_local.save(sess1, checkpoint_path, global_step=step)
	    #  scipy_io.savemat('./img/test{}.mat'.format(step),{'image_array':image_array,'masks':masks}) 	
if __name__ == '__main__':
  #tf.app.run()
  main(int(sys.argv[1]))	
