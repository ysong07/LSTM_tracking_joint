import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
from data_handler_inference_Davis import *
from params_refine_local_0 import *
from G_model_glob_share_weight import *
from G_model_local_refine import *
import os 
import time
import sys
data_dict =np.load('../model/params.npy').item()
def feature_extract_regular(batch_frames):
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



def train(test_file_name):
  sess1 = tf.Session()
  data_handler = data_handler_(sess1,batch_size=FLAGS.batch_size,length = FLAGS.length,train_file_name = test_file_name) # get batch data

  image, mask = data_handler.Get_all()
  W = image.shape[1]
  H = image.shape[2]

  with tf.Graph().as_default():
    sess = tf.Session()
    sess2 = tf.Session()

    kernel_size_dec = []
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])



    ground_truth = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,1])       
    h_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
    c_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
    h_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
    c_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,256])
    batch_frames = tf.placeholder(tf.float32,shape = [1,224,224,3])
    feature_tf = feature_extract_regular(batch_frames)
    """ placeholder for feature extractor """
    batch_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,3])
    mask_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,64])
    output_feature_1 = feature_extract(batch_frames,mask_frames)

    no_mask_frames = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,64])
    output_feature_0 = feature_extract(batch_frames,no_mask_frames)

    input_feature = feature_extract(batch_frames,no_mask_frames)

    with tf.variable_scope('trainable_params_local') as scope:

        output_feature = tf.concat([output_feature_0,output_feature_1],axis=3)
        with tf.variable_scope('initial_0'):

          c_matrix_0_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_0_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
          c_0_0 = tf.nn.conv2d(output_feature,c_matrix_0_0,strides=[1,1,1,1], padding='SAME') + c_bias_0_0

          c_0_0 = tf.tanh(c_0_0)
          c_matrix_0_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_0_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
          c_0 = tf.nn.conv2d(c_0_0,c_matrix_0_1,strides=[1,1,1,1], padding='SAME') + c_bias_0_1

          h_matrix_0_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_0_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
          h_0_0 = tf.tanh(tf.nn.conv2d(output_feature,h_matrix_0_0,strides=[1,1,1,1], padding='SAME') + h_bias_0_0)

          h_matrix_0_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_0_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
          h_0 = tf.tanh(tf.nn.conv2d(h_0_0, h_matrix_0_1,strides=[1,1,1,1], padding='SAME') + h_bias_0_1)

        with tf.variable_scope('initial_1'):
          c_matrix_1_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_1_0 = tf.get_variable("bias_c_0",shape = [512],initializer=tf.constant_initializer(0.01))
          c_1_0 = tf.nn.conv2d(output_feature,c_matrix_1_0,strides=[1,1,1,1], padding='SAME') + c_bias_1_0

          c_1_0 = tf.tanh(c_1_0)
          c_matrix_1_1 = tf.get_variable("matrix_c_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_1_1 = tf.get_variable("bias_c_1",shape = [256],initializer=tf.constant_initializer(0.01))
          c_1 = tf.nn.conv2d(c_1_0,c_matrix_1_1,strides=[1,1,1,1], padding='SAME') + c_bias_1_1

          h_matrix_1_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,512], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_1_0 = tf.get_variable("bias_h_0",shape = [512],initializer=tf.constant_initializer(0.01))
          h_1_0 = tf.tanh(tf.nn.conv2d(output_feature,h_matrix_1_0,strides=[1,1,1,1], padding='SAME') + h_bias_1_0)

          h_matrix_1_1 = tf.get_variable("matrix_h_1", shape = [3,3,512,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_1_1 = tf.get_variable("bias_h_1",shape = [256],initializer=tf.constant_initializer(0.01))
          h_1 = tf.tanh(tf.nn.conv2d(h_1_0, h_matrix_1_1,strides=[1,1,1,1], padding='SAME') + h_bias_1_1)

        G_model = G_model_local_refine(scope="G_model",height=14,width=14,length=32,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[256,256],kernel_size_dec=kernel_size_dec,num_dec_input=[256,128,64,32],num_dec_output=[128,64,32,1],layer_num_cnn =4,initial_h_0=h_0,initial_c_0=c_0,initial_h_1=h_1,initial_c_1=c_1)
    variable_collection_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_local")
    g_predict = tf.nn.sigmoid(G_model.logits):w

    saver_local = tf.train.Saver(variable_collection_local)
    saver_local.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_refine_total_large0/model.ckpt-8000")

    try: 
      os.mkdir('./img/'+'DAVIS'+test_file_name)
    except:
      pass

    for step in xrange(200):
      t= time.time()
      if step ==0:
	
	crop_size=  224
        center = [112.0,112.0]	

	# crop first template(ground truth)
        first_template = image.squeeze()
	first_template = np.stack([cv2.resize(first_template,(224,224))])

        mask_temp = mask
        first_mask = mask
	first_mask = cv2.resize(first_mask,(224,224)).reshape(224,224,1)

        first_mask = np.stack([np.tile(first_mask,(1,1,64))]) 
	feed_dict = {batch_frames:first_template,mask_frames:first_mask}
        feature_mask = sess.run(output_feature_1,feed_dict=feed_dict)

        feed_dict = {batch_frames:first_template,no_mask_frames:np.ones((1,224,224,64))}
        feature_no_mask = sess.run(output_feature_0,feed_dict=feed_dict)

        data_handler.set_id(step+1)
        image, mask = data_handler.Get_all()

        image_temp = np.stack([image],axis=0)
	feed_dict = {batch_frames:image_temp}
        feature = sess1.run(features_tf,feed_dict=feed_dict)
	G_feed_dict = {output_feature_0: feature_no_mask, output_feature_1: feature_mask, G_model.input_features:feature}
	
	
        g_predict,g_templates,g_states,g_original_template = sess.run([G_model.predicts,G_model.templates,G_model.cell_states,G_model.original_template],feed_dict= feed_dict)
	save_file_name = './img/'+'DAVIS'+test_file_name+'/'+str(step)+'.mat'
        scipy_io.savemat(save_file_name,{'image':image,'mask':mask,'predict':g_predict,'g_templates':g_original_template,'g_template_predict':g_templates})
	
      elif step<=5:
        crop_size=  224
        center = [112.0,112.0]
       	data_handler.set_id(step+1)
        image, mask = data_handler.Get_all()
		
	input_image = np.stack([image],axis=0)

        feed_dict ={c_0:g_states[0,:,:,:,:,0:256].reshape(FLAGS.batch_size,14,14,256),h_0:g_states[0,:,:,:,:,256:512].reshape(FLAGS.batch_size,14,14,256),c_1:g_states[1,:,:,:,:,0:256].reshape(FLAGS.batch_size,14,14,256),h_1:g_states[1,:,:,:,:,256:512].reshape(FLAGS.batch_size,14,14,256),G_model.center:center[::-1],G_model.crop_size:np.asarray(crop_size).reshape(1),G_model.input_frame:image}
        g_predict,g_templates,g_states,g_original_template = sess.run([G_model.predicts,G_model.templates,G_model.cell_states,G_model.original_template],feed_dict= feed_dict)

        save_file_name = './img/'+'DAVIS'+test_file_name+'/'+str(step)+'.mat'
	scipy_io.savemat(save_file_name,{'image':image,'mask':mask,'predict':g_predict,'g_templates':g_original_template,'g_template_predict':g_templates})


      else:
	crop_size=  224
        center = [112.0,112.0]
        data_handler.set_id(step+1)
        image, mask = data_handler.Get_all()
        input_image = np.stack([image],axis=0)

	feed_dict ={c_0:g_states[0,:,:,:,:,0:256].reshape(FLAGS.batch_size,14,14,256),h_0:g_states[0,:,:,:,:,256:512].reshape(FLAGS.batch_size,14,14,256),c_1:g_states[1,:,:,:,:,0:256].reshape(FLAGS.batch_size,14,14,256),h_1:g_states[1,:,:,:,:,256:512].reshape(FLAGS.batch_size,14,14,256),G_model.center:center[::-1],G_model.crop_size:np.asarray(crop_size).reshape(1),G_model.input_frame:image}
	g_predict,g_templates,g_states,g_original_template = sess.run([G_model.predicts,G_model.templates,G_model.cell_states,G_model.original_template],feed_dict= feed_dict)

        save_file_name = './img/'+'DAVIS'+test_file_name+'/'+str(step)+'.mat'
        scipy_io.savemat(save_file_name,{'image':image,'mask':mask,'predict':g_predict,'g_templates':g_original_template,'g_template_predict':g_templates})
	#summary_writer.add_summary(g_summary, step)
      elapsed = time.time() - t
            
	
      print("time per batch is " + str(elapsed))
      print(step)
def main(argv=None):  # pylint: disable=unused-argument
  with tf.device('gpu:0') :
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)

    #with open(FLAGS.test_file_dir) as file:
    #  file_list = file.readlines()
    #  file_name = file_list[argv][0:-1]

      train(argv)

	
if __name__ == '__main__':
  #tf.app.run()	
  main(sys.argv[1])
