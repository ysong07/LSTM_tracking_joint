import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
from params_refine_global_3 import * 
from G_global_train import *
import os 
import time

def train():
  sess1 = tf.Session()

  with tf.Graph().as_default():
    kernel_size_dec = []
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])
    kernel_size_dec.append([5,5])

    ground_truth = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.length-1,224,224,1])
    first_feature = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,224,224,1])
    rest_feature_ = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.length-1,224,224,1])	    
    rest_feature  = tf.reshape(rest_feature_,shape=[FLAGS.batch_size*(FLAGS.length-1),224,224,1]) 
    with tf.variable_scope('trainable_params_glob') as scope:
      with tf.variable_scope('feature_extract') as f_scope:
        def feature_extract(batch_frames):
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
  
        output_feature_0 = feature_extract(first_feature)
	f_scope.reuse_variables()
        frame_features = feature_extract(rest_feature)
        frame_features = tf.reshape(frame_features,[FLAGS.batch_size,FLAGS.length-1,14,14,64])
      with tf.variable_scope('initial_0'):
        c_matrix_0 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        c_bias_0 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
        c_0 = tf.nn.conv2d(output_feature_0,c_matrix_0,strides=[1,1,1,1], padding='SAME') + c_bias_0

        h_matrix_0 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        h_bias_0 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
        h_0 = tf.tanh(tf.nn.conv2d(output_feature_0,h_matrix_0,strides=[1,1,1,1], padding='SAME') + h_bias_0)
      with tf.variable_scope('initial_1'):
        c_matrix_1 = tf.get_variable("matrix_c", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        c_bias_1 = tf.get_variable("bias_c",shape = [64],initializer=tf.constant_initializer(0.01))
        c_1 = tf.nn.conv2d(output_feature_0,c_matrix_1,strides=[1,1,1,1], padding='SAME') + c_bias_1

        h_matrix_1 = tf.get_variable("matrix_h", shape = [3,3,64,64], initializer = tf.random_uniform_initializer(-0.01, 0.01))
        h_bias_1 = tf.get_variable("bias_h",shape = [64],initializer=tf.constant_initializer(0.01))
        h_1 = tf.tanh(tf.nn.conv2d(output_feature_0,h_matrix_1,strides=[1,1,1,1], padding='SAME') + h_bias_1)
		
      G_model = G_model_(scope="G_model",height=14,width=14,length=FLAGS.length-1,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[3,3],kernel_num=[64,64],initial_h_0=h_0,initial_c_0=c_0,initial_h_1=h_1,initial_c_1=c_1,input_features=frame_features)

      classify_loss = tf.square(ground_truth-0.1)*(tf.square(ground_truth-G_model.logits))
    #  classify_loss = tf.multiply(ground_truth, -tf.log(0.0001+G_model.logits))*100+tf.multiply((1-ground_truth),-tf.log(1-G_model.logits+0.0001))
      classify_loss = tf.reduce_mean(classify_loss[:,4:FLAGS.length-1,:,:,:])
#      preds = tf.where(tf.equal(ground_truth,1),G_model.logits),1.0-tf.sigmoid(G_model.logits))
#      preds= preds[:,4:FLAGS.length-1,:]
     # classify_loss = tf.reduce_mean(-1.0*(1.0-preds)**2*tf.log(preds+1e-7)) 

      smooth_loss = tf.reduce_mean(tf.nn.l2_loss(2*G_model.param_list[:,3:FLAGS.length-2,:] - G_model.param_list[:,4:FLAGS.length-1,:]-G_model.param_list[:,2:FLAGS.length-3,:]))
#      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= ground_truth, logits= G_model.logits))

      loss = classify_loss*10+smooth_loss*1.0
	
      temp_op = tf.train.AdamOptimizer(FLAGS.lr)
      variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"trainable_params_glob")
      gvs = temp_op.compute_gradients(loss,var_list=variable_collection)
      capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
      train_op = temp_op.apply_gradients(capped_gvs)
      scope.reuse_variables()
 
      entropy_loss_summary = tf.summary.scalar('overall_loss',loss)
      classify_loss_summary = tf.summary.scalar('classify_loss',classify_loss)
      smooth_loss_summary = tf.summary.scalar('smooth_loss',smooth_loss)		
      summary = tf.summary.merge([entropy_loss_summary,classify_loss_summary,smooth_loss_summary])

    sess = tf.Session()
    init = tf.initialize_all_variables()

    sess.run(init)

    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    """saver """
    saver = tf.train.Saver(tf.all_variables())
    saver_load=tf.train.Saver(tf.all_variables())
#    saver_load.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_resnet_globe/checkpoints/iter_2/test_new_no_smooth_share_feature_weight_0/model.ckpt-27000")
    saver_load.restore(sess,"/scratch/ys1297/LSTM_tracking/source_cross_vgg_refine/checkpoints_1/test_refine_global_3/model.ckpt-12001")
    temp_file = open(FLAGS.train_file_dir,'r')
    file_list = temp_file.readlines()
    file_id = 0

    for step in xrange(8001):

      print file_list[file_id][0:-1]
      try:

	file = h5py.File('/scratch/ys1297/LSTM_tracking/data_global_refine/'+file_list[file_id][0:-1],'r')
	id = np.arange(file['valid_num'][0])
	np.random.shuffle(id)
	id = id[0]
	true_mask = np.stack([file['true_mask'][id]])
	predict_mask = np.stack([file['predict_mask'][id]])
	ratio = np.asarray(file['ratio'][0]).reshape(1,1)
	file.close()
	temp_mask = true_mask/255.0
	temp_mask[temp_mask >0.5] =0.99
	temp_mask[temp_mask<=0.5] = 0.01
	G_feed_dict = {rest_feature_:predict_mask[:,0:31,:,:,:],G_model.ratio:ratio,ground_truth:temp_mask[:,1:32,:,:,:],first_feature:true_mask[:,0,:,:,:]*1.00}
    #    G_feed_dict = {rest_feature_:masks[:,0:masks.shape[1]-1,:,:,:], G_model.ratio:ratio,ground_truth:masks[:,1:masks.shape[1],:,:,:],first_feature:masks[:,0,:,:,:]}
        g_summary,_,g_loss,g_logits,g_transform = sess.run([summary,train_op,loss,G_model.logits,G_model.param_list],feed_dict= G_feed_dict)
	t = time.time()
#	g_transform = sess.run([G_model.param_list],feed_dict = G_feed_dict)
        summary_writer.add_summary(g_summary, step)
        elapsed = time.time() - t
	print elapsed
      except:
	elapsed = 0
	g_loss=0 
      print("time per batch is " + str(elapsed)," loss : "+str(g_loss))
      print(step) 
      file_id = file_id+1
#      g_predict,g_transform = sess.run([tf.sigmoid(G_model.logits),G_model.param_list],feed_dict = G_feed_dict)
#      scipy_io.savemat('test_{}.mat'.format(step),{'logits':g_predict,'masks':temp_mask[:,1:32,:,:,:],'predict':predict_mask[:,0:32,:,:,:]})	
      if file_id >= len(file_list):
          file_id = 0	
      if step %1000==0:
	checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step+12002)
def main(argv=None):  # pylint: disable=unused-argument
  with tf.device('gpu:0') :
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
	
if __name__ == '__main__':
  tf.app.run()	
