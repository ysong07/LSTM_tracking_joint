import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
from params_refine_STN import * 
from G_model_ST import *
import os 
import time
#from mpi4py import MPI
import cv2
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
      #if tf.gfile.Exists(FLAGS.train_dir):
      #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
      #tf.gfile.MakeDirs(FLAGS.train_dir) 

    
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
	transform_GT = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,32,2])
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


        variable_collection_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_local")
        variable_collection_local_STN = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_local/spacial_transform")
        sub_list = [item for item in variable_collection_local if item not in variable_collection_local_STN]
	
	class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= ground_truth, logits= G_model.logits))
        predict_img = tf.reshape(tf.sigmoid(G_model.logits),[32,224,224,1])
	total_loss = tf.reduce_mean(tf.image.total_variation(predict_img))
	STN_loss = tf.reduce_mean(tf.nn.l2_loss(G_model.transform_params+ transform_GT))
	loss = class_loss*0.00 + tf.minimum(total_loss,1000) *0.000+STN_loss

	temp_op = tf.train.AdamOptimizer(FLAGS.lr)

        gvs = temp_op.compute_gradients(loss,var_list=variable_collection_local_STN)
        capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
        train_op = temp_op.apply_gradients(capped_gvs)
  
        entropy_loss_summary = tf.summary.scalar('entropy_loss',class_loss)
	total_loss_summary = tf.summary.scalar('total_variation_loss',total_loss)
	STN_loss_summary = tf.summary.scalar('STN_loss',STN_loss)

        summary = tf.summary.merge([entropy_loss_summary,total_loss_summary,STN_loss_summary])
	init = tf.initialize_all_variables()
	sess1.run(init)

	saver_local = tf.train.Saver(sub_list)
        saver_local.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_refine_total_large2/model.ckpt-8000")
	saver_local = tf.train.Saver(variable_collection_local)
	#saver_local.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_2/model.ckpt-34000")
#	saver_local.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_5/test_2_STN_tanh_test/model.ckpt-6500")	
	temp_file = open(FLAGS.train_file_dir,'r')
	file_list = temp_file.readlines()
	file_id = 0
	
	for step in range(40001):
	  print file_list[file_id][0:-1]
	  try:
            file = h5py.File('/scratch/ys1297/LSTM_tracking/data_local_vgg_refine_2/'+file_list[file_id][0:-1],'r')	
	    id = np.arange(file['valid_num'][0])
	    np.random.shuffle(id)

	    id = id[0]
	    #feature = np.stack([file['feature'][id]])
	    feature_mask = np.stack([file['feature_mask'][id]])
	    feature_no_mask = np.stack([file['feature_no_mask'][id]])
	    image = np.stack([file['image'][id]])
	    masks = np.stack([file['mask'][id]])/255.0
	    masks[masks >0.5] = 0.99
	    masks[masks<=0.5] = 0.0    
	    file.close()	  

	    contaminate_list = np.arange(len(file_list)).tolist()
	    contaminate_list.pop(file_id)
	    contaminate_list = np.asarray(contaminate_list)
	    np.random.shuffle(contaminate_list)
	    contaminate_id = contaminate_list[0]
	    file_C = h5py.File('/scratch/ys1297/LSTM_tracking/data_local_vgg_refine_2/'+file_list[contaminate_id][0:-1],'r')
            id = np.arange(file_C['valid_num'][0])
            np.random.shuffle(id)

            id = id[0]
            image_C = np.stack([file_C['image'][id]])
            file_C.close()

 
	    image_temp = np.zeros((32,224,224,3))
	    masks_temp = np.zeros((1,32,224,224,1))
	    transform_params_gt = []
	    for image_id in range(32):
	      if np.random.random(1)>0.7:
                rand_size = np.random.rand(1)*0.05+0.95
                shift = np.random.randint(-20,20,size=2)
		image_temp[image_id,:,:,:] = image_C[0,image_id,:,:,:]
		masks_temp[0,image_id,:,:,0] = np.zeros((224,224))
	      else:
	        rand_size = np.random.rand(1)*0.05+0.95
	        shift = np.random.randint(-20,20,size=2)
	        M = np.float32([[rand_size[0],0,shift[0]],[0,rand_size[0],shift[1]]])
	        #M = np.float32([[1,0,0],[0,1,0]])
		image_temp[image_id,:,:,:] = cv2.warpAffine(image[:,image_id,:,:,:].squeeze(),M,(224,224))
	        masks_temp[0,image_id,:,:,0] = cv2.warpAffine(masks[:,image_id,:,:,:].squeeze(),M,(224,224))
	      transform_params_gt.append(shift/224.0)	
            masks = masks_temp
	    #feed_dict = {batch_frames:image_temp}
	    #feature = sess1.run(features_tf,feed_dict=feed_dict)
	    #feature = np.stack([feature]) 
				
	    t= time.time()
#	    G_feed_dict = {output_feature_0: feature_no_mask, output_feature_1: feature_mask, ground_truth:masks,G_model.input_features:feature}
 	    G_feed_dict = {output_feature_0: feature_no_mask, output_feature_1: feature_mask, ground_truth:masks,batch_frames:image_temp,transform_GT:np.asarray([transform_params_gt])}

	#    g_predict,tranform_feature,no_transform_feature,transform_params= sess1.run([predict_img,G_model.transform_feature,f_0,G_model.transform_params], feed_dict= G_feed_dict)
	    #scipy_io.savemat('test_STN_'+str(file_id)+'.mat',{'image':image_temp,'masks':masks_temp,'predict':g_predict})
	 #   scipy_io.savemat('test_STN_'+str(file_id)+'.mat',{'image':image_temp,'masks':masks_temp,'predict':g_predict,'feature':no_transform_feature,'transform_feature':tranform_feature,'transform_params':transform_params})
	    g_summary,_,g_loss,transform_params = sess1.run([summary,train_op,loss,G_model.transform_params],feed_dict= G_feed_dict)
	    print np.squeeze(transform_params)
	    summary_writer.add_summary(g_summary, step)
	    elapsed = time.time() - t
          except:
	    elapsed = 0
          print("time per batch is " + str(elapsed))
          print(step)
	  file_id = file_id+1
	  if file_id >= len(file_list):
	    file_id = 0
          if step %2000==0:
	    g_predict,tranform_feature,no_transform_feature,transform_params= sess1.run([predict_img,G_model.transform_feature,f_0,G_model.transform_params], feed_dict= G_feed_dict)
            scipy_io.savemat('test.mat',{'image':image_temp,'masks':masks_temp,'predict':g_predict,'feature':no_transform_feature,'transform_feature':tranform_feature,'transform_params':transform_params})
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver_local.save(sess1, checkpoint_path, global_step=step)		
	
if __name__ == '__main__':
  tf.app.run()	
