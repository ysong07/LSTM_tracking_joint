import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
from params_refine_local_1 import * 
from G_model_local_refine import *
import os 
import time
#from mpi4py import MPI


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
      if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir) 

    
      with tf.Graph().as_default():
	sess1 = tf.Session()

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
  
          G_model = G_model_local_refine(scope="G_model",height=14,width=14,length=32,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[128,128],kernel_size_dec=kernel_size_dec,num_dec_input=[128,64,32,16],num_dec_output=[64,32,16,1],layer_num_cnn =4,initial_h_0=h_0,initial_c_0=c_0,initial_h_1=h_1,initial_c_1=c_1)


        variable_collection_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_local")

        saver_local = tf.train.Saver(variable_collection_local)
        #saver_local.restore(sess1, "/scratch/ys1297/LSTM_tracking/source_cross_vgg_local/checkpoints/local_0/cross0/model.ckpt-86000")
        saver_local.restore(sess1,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints/test_refine_local_1/model.ckpt-20000")
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= ground_truth, logits= G_model.logits))
  
        temp_op = tf.train.AdamOptimizer(FLAGS.lr)

        gvs = temp_op.compute_gradients(loss,var_list=variable_collection_local)
        capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
        train_op = temp_op.apply_gradients(capped_gvs)
  
        entropy_loss_summary = tf.summary.scalar('entropy_loss',loss)
        summary = tf.summary.merge([entropy_loss_summary])
	init = tf.initialize_all_variables()
	sess1.run(init)

	
	temp_file = open(FLAGS.train_file_dir,'r')
	file_list = temp_file.readlines()
	file_id = 0
	for step in range(40001):
	  print file_list[file_id][0:-1]
	  try:
            file = h5py.File('/scratch/ys1297/LSTM_tracking/data_local_vgg_refine_1/'+file_list[file_id][0:-1],'r')	
	    id = np.arange(file['valid_num'][0])
	    np.random.shuffle(id)

	    id = id[0]
	    feature = np.stack([file['feature'][id]])
	    feature_mask = np.stack([file['feature_mask'][id]])
	    feature_no_mask = np.stack([file['feature_no_mask'][id]])
	    image = np.stack([file['image'][id]])
	    masks = np.stack([file['mask'][id]])/255.0
	    masks[masks >0.5] = 0.99
	    masks[masks<=0.5] = 0.0    
	    file.close()	  
	    
	
	    t= time.time()
	    G_feed_dict = {output_feature_0: feature_no_mask, output_feature_1: feature_mask, ground_truth:masks,G_model.input_features:feature}
            g_summary,_,g_loss = sess1.run([summary,train_op,loss],feed_dict= G_feed_dict)
	    summary_writer.add_summary(g_summary, step)
	    elapsed = time.time() - t
          except:
	    elapsed = 0
          print("time per batch is " + str(elapsed))
          print(step)
	  file_id = file_id+1
	  if file_id >= len(file_list):
	    file_id = 0
          if step %4000==0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver_local.save(sess1, checkpoint_path, global_step=step)		
	
if __name__ == '__main__':
  tf.app.run()	
