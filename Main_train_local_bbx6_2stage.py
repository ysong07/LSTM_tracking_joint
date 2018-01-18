import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
from params_refine_bbx6 import * 
from G_model_local_refine import *
import os 
import time
import cv2
import tensorflow.contrib.layers as tcl
import sys
sys.path.insert(0,'/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/anchor_gen_bbox_reg-4')
from nets.c3d_tracknet import C3DTrackNet
from model.config import cfg
import os
import glob as glob
import pdb
import copy


data_dict =np.load('../model/params.npy').item()
def feature_extract(batch_frames):
    def avg_pool( bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool( bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(bottom, name):
        with tf.variable_scope(name):
            filt = get_conv_filter(name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu
    def get_conv_filter(name):
        return tf.constant(data_dict[name][0], name="filter")

    def get_bias(name):
        return tf.constant(data_dict[name][1], name="biases")

    def get_fc_weight( name):
        return tf.constant(data_dict[name][0], name="weights")


    VGG_MEAN = [103.939, 116.779, 123.68]

    img_shape = [1,224,224]
    rgb_scaled = batch_frames
    red = rgb_scaled[:,:,:,0]
    green = rgb_scaled[:,:,:,1]
    blue = rgb_scaled[:,:,:,2]
    bgr= tf.stack([blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2]])
    bgr= tf.transpose(bgr, [1,2,3,0])
    #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    conv1_1 = conv_layer(bgr, "conv1_1")
    conv1_2 = conv_layer(conv1_1, "conv1_2")
#    conv1_2 = tf.multiply(conv1_2,mask_frames)
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, "conv2_1")
    conv2_2 = conv_layer(conv2_1, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, "conv3_1")
    conv3_2 = conv_layer(conv3_1, "conv3_2")
    conv3_3 = conv_layer(conv3_2, "conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = conv_layer(pool3, "conv4_1")
    conv4_2 = conv_layer(conv4_1, "conv4_2")
    conv4_3 = conv_layer(conv4_2, "conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')
    return pool4

def train_net(network, max_iters=40000):
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    print('Solving...')
    train_model(sess, network, max_iters)
    print('done solving')


def _get_predicted_tubes(roi_tubes,box_deltas):
    # Apply bounding-box regression deltas
    assert roi_tubes.shape[1]==cfg.NUM_FRAMES_PER_CLIP
    assert box_deltas.shape[1]==cfg.NUM_FRAMES_PER_CLIP
    predicted_tubes = np.zeros_like(roi_tubes)
    for ii in range(cfg.NUM_FRAMES_PER_CLIP):
        if cfg.useNormalizedTrans:
            pred_boxes = bbox_transform_inv_normalized(roi_tubes[:,ii,:], box_deltas[:,ii,:])
        else:
            pred_boxes = bbox_transform_inv(roi_tubes[:,ii,:], box_deltas[:,ii,:])
            pred_boxes = clip_boxes(pred_boxes)
            predicted_tubes[:,ii,:] = pred_boxes[:,4:] #1st background, then class 1
    return predicted_tubes

def test_net(net,train_model):
  sess = tf.Session(config=tfconfig)
  net.create_architecture(sess, 'TEST', 2,tag=args.tag, anchor_scales=cfg.ANCHOR_SCALES)
  saver = tf.train.Saver()
  saver.restore(sess, train_model)
  print('Loaded network {:s}'.format(train_model))
  test_model(sess,net)


def test_model(sess,net):
  blobs = {}
  blobs['input_fea'] = np.zeros((1,14,14,128))
  blobs['im_info'] = np.array([[224.,224.,1.]])

  proposals_tube,bbox_pred,scores,pos_score_map = net.test_image(sess, blobs['data'], blobs['im_info'])
  if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      pred_tubes = _get_predicted_tubes(proposals_tube,bbox_pred)
  return proposals_tube, pred_tubes, scores, pos_score_map


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

   
      	

def main(argv=None):  # pylint: disable=unused-argument
#      if tf.gfile.Exists(FLAGS.train_dir):
#        tf.gfile.DeleteRecursively(FLAGS.train_dir)
#      tf.gfile.MakeDirs(FLAGS.train_dir) 

    
      with tf.Graph().as_default():
	sess1 = tf.Session()

	batch_frames= tf.placeholder(tf.float32,shape=[32,224,224,3])
	features_tf = feature_extract(batch_frames)
	summary_writer = tf.summary.FileWriter(FLAGS.train_dir)	


        kernel_size_dec = []
        kernel_size_dec.append([5,5])
        kernel_size_dec.append([5,5])
        kernel_size_dec.append([5,5])
        kernel_size_dec.append([5,5])
	""" building training graph """
        output_feature_0 = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,512])
        output_feature_1 = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,512])  ## all object feature
        ground_truth = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,32,224,224,1])
        with tf.variable_scope('trainable_params_local') as scope:
  
          output_feature = tf.concat([output_feature_0,output_feature_1],axis=3)
          with tf.variable_scope('feature_mapping'):
            f_matrix_0_0 = tf.get_variable("matrix_f_0", shape = [3,3,512,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            f_bias_0_0 = tf.get_variable("bias_f_0",shape = [512],initializer=tf.constant_initializer(0.01))
            f_0_0 = tf.nn.conv2d(features_tf,f_matrix_0_0,strides=[1,1,1,1], padding='SAME') + f_bias_0_0
            f_0_0 = tf.nn.relu(f_0_0)
            f_matrix_0_1 = tf.get_variable("matrix_f_1", shape = [3,3,512,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            f_bias_0_1 = tf.get_variable("bias_f_1",shape = [512],initializer=tf.constant_initializer(0.01))
            f_0 = tf.nn.conv2d(f_0_0,f_matrix_0_1,strides=[1,1,1,1], padding='SAME') + f_bias_0_1
            f_0 = tf.tanh(f_0)

          with tf.variable_scope('initial_0'):
  
            c_matrix_0_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_0_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
            c_0_0 = tf.nn.conv2d(output_feature,c_matrix_0_0,strides=[1,1,1,1], padding='SAME') + c_bias_0_0
  
            c_0_0 = tf.nn.relu(c_0_0)
            c_matrix_0_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_0_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
            c_0 = tf.tanh(tf.nn.conv2d(c_0_0,c_matrix_0_1,strides=[1,1,1,1], padding='SAME') + c_bias_0_1)
  
            h_matrix_0_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_0_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
            h_0_0 = tf.nn.relu(tf.nn.conv2d(output_feature,h_matrix_0_0,strides=[1,1,1,1], padding='SAME') + h_bias_0_0)
  
            h_matrix_0_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_0_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
            h_0 = tf.tanh(tf.nn.conv2d(h_0_0, h_matrix_0_1,strides=[1,1,1,1], padding='SAME') + h_bias_0_1)
  
          with tf.variable_scope('initial_1'):
            c_matrix_1_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_1_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
            c_1_0 = tf.nn.conv2d(output_feature,c_matrix_1_0,strides=[1,1,1,1], padding='SAME') + c_bias_1_0
  
            c_1_0 = tf.nn.relu(c_1_0)
            c_matrix_1_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            c_bias_1_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
            c_1 = tf.tanh(tf.nn.conv2d(c_1_0,c_matrix_1_1,strides=[1,1,1,1], padding='SAME') + c_bias_1_1)
  
            h_matrix_1_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_1_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
            h_1_0 = tf.nn.relu(tf.nn.conv2d(output_feature,h_matrix_1_0,strides=[1,1,1,1], padding='SAME') + h_bias_1_0)
  
            h_matrix_1_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
            h_bias_1_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
            h_1 = tf.tanh(tf.nn.conv2d(h_1_0, h_matrix_1_1,strides=[1,1,1,1], padding='SAME') + h_bias_1_1)
  
          G_model = G_model_local_refine(scope="G_model",height=14,width=14,length=32,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[256,256],kernel_size_dec=kernel_size_dec,num_dec_input=[256,128,64,32],num_dec_output=[128,64,32,1],layer_num_cnn =4,initial_h_0=h_0,initial_c_0=c_0,initial_h_1=h_1,initial_c_1=c_1,input_feature=tf.stack([f_0]))

	LSTM_feature = G_model.lstm_output[0,:,:,:,:]
        variable_collection_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_local")
	
	
        with tf.variable_scope('bbx_params') as scope:

          network = C3DTrackNet(batch_size=1)
          layers = network.create_architecture(sess1, 'TRAIN', 2,
                  tag='default', anchor_scales=cfg.ANCHOR_SCALES)

        loss = layers['total_loss']
        variable_collection_bbx = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"bbx_params")
        saver_bbx = tf.train.Saver(variable_collection_bbx)
	#loss = tf.reduce_mean(tf.nn.l2_loss(ground_truth- fc2)) 
        #class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= ground_truth, logits= G_model.logits))  

        temp_op = tf.train.AdamOptimizer(FLAGS.lr)

        local_bbx = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"bbx_params")

        gvs = temp_op.compute_gradients(loss,var_list=local_bbx)
        capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
        train_op = temp_op.apply_gradients(capped_gvs)

        entropy_loss_summary = tf.summary.scalar('l2_loss',loss)
        summary = tf.summary.merge([entropy_loss_summary])
        init = tf.initialize_all_variables()
        sess1.run(init)
#	saver_bbx.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_refine_bbx0/model.ckpt-4000")
        saver_local = tf.train.Saver(variable_collection_local)
        saver_local.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_refine_total_large6/model_keep.ckpt-24")
#	saver_local.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_2/model.ckpt-34000")
#	saver_local.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_more_contaminate_large2/model.ckpt-24000")	
	temp_file = open(FLAGS.train_file_dir,'r')
	file_list = temp_file.readlines()
	file_id = 0

	for step in range(20001):
          print file_list[file_id][0:-1]
          try:
            file = h5py.File('/scratch/ys1297/LSTM_tracking/data_local_vgg_refine_6/'+file_list[file_id][0:-1],'r')
            id = np.arange(file['valid_num'][0])
            np.random.shuffle(id)

            id = id[0]
#           feature = np.stack([file['feature'][id]])
            feature_mask = np.stack([file['feature_mask'][id]])
            feature_no_mask = np.stack([file['feature_no_mask'][id]])
            image = file['image'][id]
            
            bbx = np.stack(file['bbx'][id])/224.0

            file.close()
            bbx =  bbx[:,0:4]

            t= time.time()
            G_feed_dict = {output_feature_0: feature_no_mask, output_feature_1: feature_mask, batch_frames: image}

	    feature = sess1.run(LSTM_feature, feed_dict= G_feed_dict)
            tmp =copy.copy(bbx)
            bbx[:,0]= tmp[:,0] - tmp[:,2]/2.0
            bbx[:,1]= tmp[:,1] - tmp[:,3]/2.0
            bbx[:,2]= tmp[:,0] + tmp[:,2]/2.0
            bbx[:,3]= tmp[:,1] + tmp[:,3]/2.0

            bbx[bbx <=0.0] = 0.001
            bbx[bbx>=1.0] = 0.999
            bbx = bbx*224.0
            for i in range(32):
              blobs = {}
              blobs['input_fea'] = feature[i:i+1]

              blobs['gt_boxes'] = np.asarray([bbx[i].tolist()+[1.0]]).reshape(1,1,5)
              blobs['im_info'] = np.array([[224.,224.,1.]])
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = network.train_step(sess1, blobs, train_op)
              print rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss

            elapsed = time.time() - t
          except:
            elapsed = 0
          print("time per batch is " + str(elapsed))
          print(step)
          file_id = file_id +1
          if file_id >= len(file_list):
            file_id = 0
          if step %4000==0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver_bbx.save(sess1, checkpoint_path, global_step=step)	
	
	
if __name__ == '__main__':
  tf.app.run()	
