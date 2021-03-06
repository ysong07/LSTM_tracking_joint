import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
from data_handler_inference import *
from params_refine_local_8 import *
from G_model_glob_share_weight import *
from G_inference import *
import os 
import time
import sys
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

#data_dict =np.load('../model/params.npy').item()
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

def get_center(mask):
  yv,xv = np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0])) 
  xv = np.float32(xv)
  yv = np.float32(yv)
   
  center_x,center_y = np.int32(np.sum(xv*mask)/mask.sum()),np.int32(np.sum(yv*mask)/mask.sum())
  center = [center_y,center_x]
  return center

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



#    output_feature_0 = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,512])
#    output_feature_1 = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,512])
    ground_truth = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,1])       
    h_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])
    c_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])
    h_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])
    c_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,128])


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

          c_matrix_0_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_0_0 = tf.get_variable("bias_c_0",shape = [256],initializer=tf.constant_initializer(0.01))
          c_0_0 = tf.nn.conv2d(output_feature,c_matrix_0_0,strides=[1,1,1,1], padding='SAME') + c_bias_0_0

          c_0_0 = tf.tanh(c_0_0)
          c_matrix_0_1 = tf.get_variable("matrix_c_1", shape = [3,3,256,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_0_1 = tf.get_variable("bias_c_1",shape = [128],initializer=tf.constant_initializer(0.01))
          c_0 = tf.nn.conv2d(c_0_0,c_matrix_0_1,strides=[1,1,1,1], padding='SAME') + c_bias_0_1

          h_matrix_0_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_0_0 = tf.get_variable("bias_h_0",shape = [256],initializer=tf.constant_initializer(0.01))
          h_0_0 = tf.tanh(tf.nn.conv2d(output_feature,h_matrix_0_0,strides=[1,1,1,1], padding='SAME') + h_bias_0_0)

          h_matrix_0_1 = tf.get_variable("matrix_h_1", shape = [3,3,256,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_0_1 = tf.get_variable("bias_h_1",shape = [128],initializer=tf.constant_initializer(0.01))
          h_0 = tf.tanh(tf.nn.conv2d(h_0_0, h_matrix_0_1,strides=[1,1,1,1], padding='SAME') + h_bias_0_1)

        with tf.variable_scope('initial_1'):
          c_matrix_1_0 = tf.get_variable("matrix_c_0", shape = [3,3,512*2,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_1_0 = tf.get_variable("bias_c_0",shape = [256],initializer=tf.constant_initializer(0.01))
          c_1_0 = tf.nn.conv2d(output_feature,c_matrix_1_0,strides=[1,1,1,1], padding='SAME') + c_bias_1_0

          c_1_0 = tf.tanh(c_1_0)
          c_matrix_1_1 = tf.get_variable("matrix_c_1", shape = [3,3,256,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          c_bias_1_1 = tf.get_variable("bias_c_1",shape = [128],initializer=tf.constant_initializer(0.01))
          c_1 = tf.nn.conv2d(c_1_0,c_matrix_1_1,strides=[1,1,1,1], padding='SAME') + c_bias_1_1

          h_matrix_1_0 = tf.get_variable("matrix_h_0", shape = [3,3,512*2,256], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_1_0 = tf.get_variable("bias_h_0",shape = [256],initializer=tf.constant_initializer(0.01))
          h_1_0 = tf.tanh(tf.nn.conv2d(output_feature,h_matrix_1_0,strides=[1,1,1,1], padding='SAME') + h_bias_1_0)

          h_matrix_1_1 = tf.get_variable("matrix_h_1", shape = [3,3,256,128], initializer = tf.random_uniform_initializer(-0.01, 0.01))
          h_bias_1_1 = tf.get_variable("bias_h_1",shape = [128],initializer=tf.constant_initializer(0.01))
          h_1 = tf.tanh(tf.nn.conv2d(h_1_0, h_matrix_1_1,strides=[1,1,1,1], padding='SAME') + h_bias_1_1)

        G_model = G_model_(scope="G_model",height=14,width=14,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[128,128],kernel_size_dec=kernel_size_dec,num_dec_input=[128,64,32,16],num_dec_output=[64,32,16,1],layer_num_cnn =4,initial_h_0=h_0,initial_c_0=c_0,initial_h_1=h_1,initial_c_1=c_1,img_height=W,img_width=H) 

    variable_collection_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_local")
     

    saver_local = tf.train.Saver(variable_collection_local)
    #saver_local.restore(sess, "/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints/test_refine_local_0/model.ckpt-20000")
    #saver_local.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_local/checkpoints/local_0/cross1/model.ckpt-216000")
#    saver_local.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_1/test_refine_shuffle_0/model.ckpt-40000")	
    saver_local.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_1/test_refine_total_8/model.ckpt-40000")

    with tf.variable_scope('bbx_params') as scope:	
      input_ = G_model.lstm_output[0,0:1,:,:,:]
      input_ = tcl.flatten(input_)
      fc0 = tc.layers.fully_connected(
                input_, 4096,
                weights_initializer=tf.random_normal_initializer(stddev=0.002),
                activation_fn=tf.identity
                )
      fc0 = tf.maximum(0.2 * fc0,fc0)
      fc1 = tc.layers.fully_connected(
                fc0, 1024,
            )
      fc1 = tf.maximum(0.2 * fc1,fc1)
      fc2 = tc.layers.fully_connected(
                fc1, 4,
                weights_initializer=tf.random_normal_initializer(stddev=0.002),
                activation_fn=tf.identity
           )
      fc2 = tf.nn.sigmoid(fc2)
    variable_collection_bbx = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"bbx_params") 
    saver_bbx = tf.train.Saver(variable_collection_bbx)
    saver_bbx.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_2/bbx_8/model.ckpt-40000")


    #saver_local.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_local/checkpoints/local_0/cross0/model.ckpt-86000")
    ground_truth_glob = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.length-1,224,224,1])
   
    glob_h_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
    glob_c_0  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
    glob_h_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
    glob_c_1  = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,14,14,64])
     
    with tf.variable_scope('trainable_params_glob') as scope:
      first_feature = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,1])
      rest_feature = tf.placeholder(tf.float32,shape=[FLAGS.batch_size*(FLAGS.length-1),224,224,1])
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

        glob_feature_0 = feature_extract_mask(first_feature)
        f_scope.reuse_variables()
        frame_features = feature_extract_mask(rest_feature)
      with tf.variable_scope('initial_0'):
        c_matrix_0 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        c_bias_0 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
        glob_c_0 = tf.nn.conv2d(glob_feature_0,c_matrix_0,strides=[1,1,1,1], padding='SAME') + c_bias_0

        h_matrix_0 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        h_bias_0 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
        glob_h_0 = tf.tanh(tf.nn.conv2d(glob_feature_0,h_matrix_0,strides=[1,1,1,1], padding='SAME') + h_bias_0)

      with tf.variable_scope('initial_1'):
        c_matrix_1 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        c_bias_1 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
        glob_c_1 = tf.nn.conv2d(glob_feature_0,c_matrix_1,strides=[1,1,1,1], padding='SAME') + c_bias_1

        h_matrix_1 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        h_bias_1 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
        glob_h_1 = tf.tanh(tf.nn.conv2d(glob_feature_0,h_matrix_1,strides=[1,1,1,1], padding='SAME') + h_bias_1)

      G_model_G = G_model_glob(scope="G_model",height=14,width=14,length=FLAGS.length-1,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[64,64],initial_h_0=glob_h_0,initial_c_0=glob_c_0,initial_h_1=glob_h_1,initial_c_1=glob_c_1,input_features=frame_features)


    variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_glob")
    """saver """
    #saver = tf.train.Saver(tf.all_variables())
    #saver_load=tf.train.Saver(tf.all_variables())
    #saver = tf.train.import_meta_graph("/scratch/ys1297/LSTM_tracking/source_crop/checkpoints/test_new_no_smooth/model.ckpt-9000.meta")
    #saver.restore(sess,"/scratch/ys1297/LSTM_tracking/source_crop/checkpoints/test_new_no_smooth/model.ckpt-9000")
    saver = tf.train.Saver(variable_collection)
    saver.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_resnet_globe/checkpoints/iter_2/test_new_no_smooth_share_feature_weight_8/model.ckpt-27000")
 
    
#    saver.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_1/test_refine_global_0/model.ckpt-1000")

    for step in xrange(2000):
      t= time.time()
      if step ==0:
        masks_resize = cv2.resize(mask[0,:,:,0],(224,224)).reshape(1,224,224,1)
	image_resize = cv2.resize(image[0,:,:,:],(224,224)).reshape(1,224,224,3)
	
	crop_size = 4*np.int(np.sqrt((mask[0,:,:,0]>0.1).sum()))
	crop_size = min(crop_size,224*2)
	crop_size = max(crop_size,112)

	center = get_center(mask[0,:,:,0].squeeze())
	ratio = np.array([float(mask.shape[1])/mask.shape[2]]).reshape(1,1)
	G_glob_dict = {rest_feature:masks_resize, G_model_G.ratio:ratio,first_feature:masks_resize}
	glob_frame,glob_cell,glob_crop_size,glob_center = sess.run([G_model_G.logits,G_model_G.cell_state,G_model_G.crop_size,G_model_G.center],feed_dict= G_glob_dict)

	# crop first template(ground truth)
        first_template = crop_image_include_coundary(np.zeros((crop_size,crop_size,3)),image.squeeze(),(center[1],center[0]))
        first_template = cv2.resize(first_template,(224,224)).reshape(FLAGS.batch_size,224,224,3)

        mask_temp = mask[0,:,:,0:1]>0
        first_mask = crop_image_include_coundary(np.zeros((crop_size,crop_size,1)),mask_temp,(center[1],center[0]))
        first_mask = cv2.resize(first_mask,(224,224)).reshape(224,224,1)

        first_mask = np.tile(first_mask,(1,1,64)).reshape(FLAGS.batch_size,224,224,64)        
	feed_dict = {batch_frames:first_template,mask_frames:first_mask}
        feature_mask = sess.run(output_feature_1,feed_dict=feed_dict)

        feed_dict = {batch_frames:first_template,no_mask_frames:np.ones((1,224,224,64))}
        feature_no_mask = sess.run(output_feature_0,feed_dict=feed_dict)

        data_handler.set_id(step+1)
        image, mask = data_handler.Get_all()

        input_image = np.stack([image],axis=0)

        feed_dict ={G_model.input_frame:image,output_feature_1:feature_mask,output_feature_0:feature_no_mask,G_model.center:center[::-1],G_model.crop_size:np.asarray(crop_size).reshape(1)}
	g_predict,g_templates,g_states,g_original_template,bbx_predict ,input_np= sess.run([G_model.predicts,G_model.templates,G_model.cell_states,G_model.original_template,fc2,input_],feed_dict= feed_dict)
	save_file_name = './bbx/'+re.sub('origin.h5','',test_file_name)+'_'+str(step)+'.mat'
        scipy_io.savemat(save_file_name,{'glob_frame':glob_frame,'image':image,'mask':mask,'predict':g_predict,'g_templates':g_original_template,'g_template_predict':g_templates,'bbx':bbx_predict})
	
      elif step<=5:
	
	  
        crop_size_0 = 4*np.int(np.sqrt((g_predict.squeeze()>0.1).sum()))
        center_0 = get_center(g_predict.squeeze()) 

	masks_resize = np.zeros([1,224,224,1])
        temp = cv2.resize(g_predict,(224,224)).reshape(1,224,224,1)
        #masks_resize = temp*255.0
        masks_resize[temp>=0.5]=255.0	 
	ratio = np.array([float(mask.shape[1])/mask.shape[2]]).reshape(1,1)

	data_handler.set_id(step+1)
        image, mask = data_handler.Get_all()
	G_glob_dict = {rest_feature:masks_resize, G_model_G.ratio:ratio,glob_c_0:glob_cell[0:1,0,:,:,0:64],glob_h_0:glob_cell[0:1,0,:,:,64:128],glob_c_1:glob_cell[1:2,0,:,:,0:64],glob_h_1:glob_cell[1:2,0,:,:,64:128]}
	glob_frame,glob_cell,glob_crop_size,glob_center = sess.run([G_model_G.logits,G_model_G.cell_state,G_model_G.crop_size,G_model_G.center],feed_dict= G_glob_dict)

	crop_size_1 = int(image.shape[1]/glob_crop_size/224.0*64)*2
        #crop_size = np.asarray(glob_crop_size*64*mask.shape[1]/224)
        center_1 = np.zeros([2],dtype = 'int32')
        temp = (1-glob_center[0,0]/glob_crop_size*ratio)/2.0 ## location at 224 by 224
        center_1[0] = temp *image.shape[2]
        temp = (1-glob_center[0,1]/glob_crop_size)/2.0 ## location at 224 by 224
        center_1[1] = temp *image.shape[1]
       
	crop_size = (crop_size_0 *(6.0-step)+ crop_size_1*float(step))/6.0 
	crop_size = min(crop_size,224*2)
        crop_size = max(crop_size,112)
 	center = (np.asarray(center_0)*(6.0-step)+np.asarray(center_1)*float(step))/6.0
	center = np.int32(center).tolist()	

	input_image = np.stack([image],axis=0)

        feed_dict ={c_0:g_states[0,:,:,:,:,0:128].reshape(FLAGS.batch_size,14,14,128),h_0:g_states[0,:,:,:,:,128:256].reshape(FLAGS.batch_size,14,14,128),c_1:g_states[1,:,:,:,:,0:128].reshape(FLAGS.batch_size,14,14,128),h_1:g_states[1,:,:,:,:,128:256].reshape(FLAGS.batch_size,14,14,128),G_model.center:center[::-1],G_model.crop_size:np.asarray(crop_size).reshape(1),G_model.input_frame:image}
        g_predict,g_templates,g_states,g_original_template,bbx_predict= sess.run([G_model.predicts,G_model.templates,G_model.cell_states,G_model.original_template,fc2],feed_dict= feed_dict)

        save_file_name = './bbx/'+re.sub('origin.h5','',test_file_name)+'_'+str(step)+'.mat'
        scipy_io.savemat(save_file_name,{'glob_frame':glob_frame,'image':image,'mask':mask,'predict':g_predict,'g_templates':g_original_template,'g_template_predict':g_templates,'bbx':bbx_predict})



      else:
	#masks_resize = cv2.resize(mask[0,:,:,0],(224,224)).reshape(1,224,224,1)
	masks_resize = np.zeros([1,224,224,1])
	temp = cv2.resize(g_predict,(224,224)).reshape(1,224,224,1)
	#masks_resize = temp*255.0
        masks_resize[temp>=0.5]=255.0
	#masks_resize = cv2.resize(g_predict,(224,224)).reshape(1,224,224,1)
	
	
	data_handler.set_id(step+1)
        image, mask = data_handler.Get_all()
 
	ratio = np.array([float(mask.shape[1])/mask.shape[2]]).reshape(1,1)
        G_glob_dict = {rest_feature:masks_resize, G_model_G.ratio:ratio,glob_c_0:glob_cell[0:1,0,:,:,0:64],glob_h_0:glob_cell[0:1,0,:,:,64:128],glob_c_1:glob_cell[1:2,0,:,:,0:64],glob_h_1:glob_cell[1:2,0,:,:,64:128]}
        glob_frame,glob_cell,glob_crop_size,glob_center = sess.run([G_model_G.logits,G_model_G.cell_state,G_model_G.crop_size,G_model_G.center],feed_dict= G_glob_dict)

	old_crop_size = crop_size
	old_center = np.asarray(center)
        crop_size = int(image.shape[1]/glob_crop_size/224.0*64)*2
        crop_size = int((crop_size*0.1+old_crop_size*.9))
	crop_size = min(crop_size,224*2)
	crop_size = max(crop_size,112)

	
	center = np.zeros([2],dtype = 'int32')
        temp = (1-glob_center[0,0]/glob_crop_size*ratio)/2.0 ## location at 224 by 224
        center[0] = temp *image.shape[2]
        temp = (1-glob_center[0,1]/glob_crop_size)/2.0 ## location at 224 by 224
        center[1] = temp *image.shape[1]
	center = ((old_center*0.9+ np.asarray(center)*0.1)).tolist()

        input_image = np.stack([image],axis=0)

	feed_dict ={c_0:g_states[0,:,:,:,:,0:128].reshape(FLAGS.batch_size,14,14,128),h_0:g_states[0,:,:,:,:,128:256].reshape(FLAGS.batch_size,14,14,128),c_1:g_states[1,:,:,:,:,0:128].reshape(FLAGS.batch_size,14,14,128),h_1:g_states[1,:,:,:,:,128:256].reshape(FLAGS.batch_size,14,14,128),G_model.center:center[::-1],G_model.crop_size:np.asarray(crop_size).reshape(1),G_model.input_frame:image}
	g_predict,g_templates,g_states,g_original_template,bbx_predict = sess.run([G_model.predicts,G_model.templates,G_model.cell_states,G_model.original_template,fc2],feed_dict= feed_dict)

	save_file_name = './bbx/'+re.sub('origin.h5','',test_file_name)+'_'+str(step)+'.mat'
      	scipy_io.savemat(save_file_name,{'glob_frame':glob_frame,'image':image,'mask':mask,'predict':g_predict,'g_templates':g_original_template,'g_template_predict':g_templates,'bbx':bbx_predict})
        
 
      #summary_writer.add_summary(g_summary, step)
      elapsed = time.time() - t
            
	
      print("time per batch is " + str(elapsed))
      print(step)
def main(argv=None):  # pylint: disable=unused-argument
  with tf.device('gpu:0') :
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)

    with open(FLAGS.test_file_dir) as file:
      file_list = file.readlines()
      file_name = file_list[argv][0:-1]

      train(file_name)

	
if __name__ == '__main__':
  #tf.app.run()	
  main(int(sys.argv[1]))
