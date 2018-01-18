import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
import BasicConvLSTMCell
from spatial_transformer import transformer


class G_model_local_refine:
  def __init__ (self,scope,height,width,length,batch_size,layer_num_lstm,kernel_size,kernel_num,kernel_size_dec,num_dec_input,num_dec_output,layer_num_cnn,initial_h_0,initial_c_0,initial_h_1,initial_c_1,input_feature=None):
    self.scope = scope
    self.height = height
    self.width = width
    self.length =length 
    self.batch_size = batch_size
    self.layer_num_lstm = layer_num_lstm
    self.kernel_size = kernel_size
    self.kernel_num = kernel_num

    self.kernel_size_dec = kernel_size_dec
    self.num_dec_input = num_dec_input
    self.num_dec_output = num_dec_output
    self.layer_num_cnn  = layer_num_cnn
 
    self.initial_h_0 = initial_h_0 
    self.initial_c_0 = initial_c_0  
    self.initial_h_1 = initial_h_1  
    self.initial_c_1 = initial_c_1  
    with tf.name_scope('input'):
        if input_feature is None :
          self.input_features = tf.placeholder(tf.float32, shape=[self.batch_size, self.length, self.height, self.width,512])

        else:
          self.input_features = input_feature
    self.define_graph()
  def define_graph(self):
    with tf.variable_scope('spacial_transform') as new_scope:
	self.STN_conv_W = []
        self.STN_conv_b = []
	self.STN_W = []
	self.STN_b = []
	with tf.variable_scope("conv_0"):
	   self.STN_conv_W.append(tf.get_variable("matrix",shape =[3,3,1024,512],initializer = tf.random_uniform_initializer(-0.01, 0.01)))
           self.STN_conv_b.append(tf.get_variable("bias",shape = [512],initializer=tf.constant_initializer(0.01)))
	with tf.variable_scope("conv_1"):
           self.STN_conv_W.append(tf.get_variable("matrix",shape =[3,3,512,256],initializer = tf.random_uniform_initializer(-0.01, 0.01)))
           self.STN_conv_b.append(tf.get_variable("bias",shape = [256],initializer=tf.constant_initializer(0.01))) 	
	with tf.variable_scope("full_0"):
           self.STN_W.append(tf.get_variable("matrix",shape =[7*7*256,1024],initializer = tf.random_uniform_initializer(-0.01, 0.01)))
           self.STN_b.append(tf.get_variable("bias",shape = [1024],initializer=tf.constant_initializer(0.01)))
	with tf.variable_scope("full_1"):
	   self.STN_W.append(tf.get_variable("matrix",shape =[1024,2],initializer = tf.random_uniform_initializer(-0.001, 0.001)))
           self.STN_b.append(tf.get_variable("bias",shape = [2],initializer=tf.constant_initializer(0.0)))

	
    with tf.variable_scope(self.scope) as scope:	
      self.lstms = []
      self.lstms_state = []
      for layer_id_, kernel_, kernel_num_ in zip(xrange(self.layer_num_lstm),self.kernel_size,self.kernel_num):
	layer_name_encode = 'conv_lstm'+str(layer_id_)+'enc'
        temp_cell= BasicConvLSTMCell.BasicConvLSTMCell([self.height,self.width],[kernel_,kernel_],kernel_num_,layer_name_encode)
	if layer_id_ ==0:
	   self.lstms_state.append(tf.concat([self.initial_c_0,self.initial_h_0],3))
	else:
	   self.lstms_state.append(tf.concat([self.initial_c_1,self.initial_h_1],3)) 
      
        self.lstms.append(temp_cell)

      input_ = tf.concat([self.input_features[:,0,:,:,:],self.initial_c_0,self.initial_c_1],axis=3)
      input_ = tf.nn.conv2d(input_,self.STN_conv_W[0],strides=[1,1,1,1],padding='SAME')+self.STN_conv_b[0]
      input_ = tf.nn.relu(input_)
      input_ = tf.nn.conv2d(input_,self.STN_conv_W[1],strides=[1,1,1,1],padding='SAME')+self.STN_conv_b[1]
      input_ = tf.nn.relu(input_)
      input_ = tf.nn.max_pool(input_,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      input_ = tf.contrib.layers.flatten(input_)
      input_ = tf.matmul(input_,self.STN_W[0])+self.STN_b[0]
      input_ = tf.nn.relu(input_)
      input_ = tf.matmul(input_,self.STN_W[1])+self.STN_b[1]   

 
      self.dec_conv_W = []
      self.dec_conv_b = []	
      for layer_id_,kernel_,input_,output_ in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
        with tf.variable_scope("dec_conv_{}".format(layer_id_)):
          self.dec_conv_W.append(tf.get_variable("matrix", shape = kernel_+[output_,input_], initializer = tf.random_uniform_initializer(-0.01, 0.01)))
          self.dec_conv_b.append(tf.get_variable("bias",shape = [output_],initializer=tf.constant_initializer(0.01)))
	  
      input_ = self.input_features[:,0,:,:,:]
      for lstm_layer in range(self.layer_num_lstm):
        input_,_=self.lstms[lstm_layer](input_,self.lstms_state[lstm_layer])	

      for layer_id_,kernel_num_,num_input,num_output in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
	if layer_id_ == self.layer_num_cnn-1:
	  output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])	  
	  input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
	  input_ = tf.nn.sigmoid(input_)

	else:
	  output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])                            
          input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
	  input_ = tf.maximum(0.2 * input_, input_)

      scope.reuse_variables()	

      def forward():
        output = []
	cell_state = []
	transform_feature = []
	transform_params = []
	for frame_id in xrange(self.length):
	  #input_ = self.input_features[:,frame_id,:,:,:]
          input_ = tf.concat([self.input_features[:,frame_id,:,:,:],self.initial_c_0,self.initial_c_1],axis=3)
          input_ = tf.nn.conv2d(input_,self.STN_conv_W[0],strides=[1,1,1,1],padding='SAME')+self.STN_conv_b[0]
	  input_ = tf.nn.relu(input_)
          input_ = tf.nn.conv2d(input_,self.STN_conv_W[1],strides=[1,1,1,1],padding='SAME')+self.STN_conv_b[1]
	  input_ = tf.nn.relu(input_)
	  input_ = tf.nn.max_pool(input_,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	  input_ = tf.contrib.layers.flatten(input_)
          input_ = tf.matmul(input_,self.STN_W[0])+self.STN_b[0]
	  input_ = tf.nn.relu(input_)
          input_ = tf.matmul(input_,self.STN_W[1])+self.STN_b[1]
	 
	  input_ = tf.tanh(input_)*0.3
	  transform_params.append(input_) 
#	  input_= tf.maximum(input_,tf.ones(input_.shape)*-2.0)
#	  input_ = tf.minimum(input_,tf.ones(input_.shape)*2.0)
	  params = tf.stack([tf.ones(shape=self.batch_size),tf.zeros(shape=self.batch_size),input_[:,0],tf.zeros(shape=self.batch_size),tf.ones(shape=self.batch_size),input_[:,1]],axis=1)
	  inverse_params = tf.stack([tf.ones(shape=self.batch_size),tf.zeros(shape=self.batch_size),-input_[:,0],tf.zeros(shape=self.batch_size),tf.ones(shape=self.batch_size),-input_[:,1]],axis=1)

	  input_ = transformer(self.input_features[:,frame_id,:,:,:],params,[14,14],output_channel=512)	
	  transform_feature.append(input_)
	  for lstm_layer in range(self.layer_num_lstm):
            input_,self.lstms_state[lstm_layer]=self.lstms[lstm_layer](input_,self.lstms_state[lstm_layer])
	  #input_,self.lstms_state[0]=self.lstms[0](input_,self.lstms_state[0])
          #input_,self.lstms_state[1]=self.lstms[1](input_,self.lstms_state[1])
	
	  input_ =  transformer(input_,inverse_params,[14,14],output_channel=256)  
	  for layer_id_,kernel_num_,num_input,num_output in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
            if layer_id_ == self.layer_num_cnn-1:
              output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
              input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
              #input_ = tf.nn.sigmoid(input_)
	      output.append(input_)
            else:
              output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
              input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
              input_ = tf.maximum(0.2 * input_, input_)


        output = tf.stack(output,axis=1)	  
	return output,tf.stack(transform_feature,axis=1),tf.stack([transform_params],axis=1)

      self.logits,self.transform_feature,self.transform_params = forward() #note: output is logits need to convert to sigmoid in testing mode


    def forward_oneframe(self,input_feature):
      with tf.variable_scope(self.scope) as scope:
	scope.reuse_variables()
      	input_ = input_feature[:,:,:,:]
	for lstm_layer in range(self.layer_num_lstm):
          input_,self.lstms_state[lstm_layer]=self.lstms[lstm_layer](input_,self.lstms_state[lstm_layer])
	#input_,self.lstms_state[0]=self.lstms[0](input_,self.lstms_state[0])
        #input_,self.lstms_state[1]=self.lstms[1](input_,self.lstms_state[1])

	for layer_id_,kernel_num_,num_input,num_output in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
          if layer_id_ == self.layer_num_cnn-1:
            output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
            input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
            output= input_
          else:
            output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
            input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
            input_ = tf.maximum(0.2 * input_, input_)

      return output	
	#self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ground_truth, logits= self.logits)
	#temp_op = tf.train.AdamOptimizer(FLAGS.lr)
        #variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                 self.scope)
	#gvs = temp_op.compute_gradients(self.loss,var_list=variable_collection)
        #capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
        #self.train_op = temp_op.apply_gradients(capped_gvs)
	#
	#entropy_loss_summary = tf.summary.scalar('entropy_loss',self.loss)
	#self.summary = tf.summary.merge([entropy_loss_summary])

