import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
import BasicConvLSTMCell
from spatial_transformer import transformer
class G_model_glob:
  def __init__ (self,scope,height,width,length,batch_size,layer_num_lstm,kernel_size,kernel_num,initial_h_0,initial_c_0,initial_h_1,initial_c_1,input_features):
    self.scope = scope
    self.height = height
    self.width = width
    self.batch_size = batch_size
    self.layer_num_lstm = layer_num_lstm
    self.kernel_size = kernel_size
    self.kernel_num = kernel_num

    self.initial_h_0 = initial_h_0
    self.initial_c_0 = initial_c_0
    self.initial_h_1 = initial_h_1
    self.initial_c_1 = initial_c_1 
    self.input_features =input_features
    self.define_graph()

  def define_graph(self):
    with tf.name_scope('input'):
      #self.input_features = tf.placeholder(tf.float32, shape=[self.batch_size,self.length, 14,14,64])
      self.ratio = tf.placeholder(tf.float32,shape=[self.batch_size,1])

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
        #if layer_id_ ==0:
        #   self.lstms_state.append(temp_cell.zero_state(self.batch_size,dtype=tf.float32))
        #else:
        #   self.lstms_state.append(temp_cell.zero_state(self.batch_size,dtype=tf.float32))
	self.lstms.append(temp_cell)
      
    
      temp = np.zeros((self.batch_size,224,224,1))
      temp[:,112-32:112+32,112-32:112+32,:]=1
      self.transform_matrix = tf.constant(temp)
 
      self.STN_W = []
      self.STN_b = []	
		
      with tf.variable_scope("STN_0"):
        self.STN_W.append(tf.get_variable("matrix",shape =[14*14*64,1024],initializer = tf.random_uniform_initializer(-0.01, 0.01)))

        self.STN_b.append(tf.get_variable("bias",shape = [1024],initializer=tf.constant_initializer(0.01)))

      with tf.variable_scope("STN_1"):   
        self.STN_W.append(tf.get_variable("matrix",shape =[1025,3],initializer = tf.random_uniform_initializer(-0.01, 0.01)))
        self.STN_b.append(tf.get_variable("bias",shape = [3],initializer=tf.constant_initializer([1.0,0.0,0.0]))) 

      input_ = self.input_features[:,:,:,:] 
      #input_ = self.input_features[:,0,:,:,:]
      for lstm_layer in range(self.layer_num_lstm):
        input_,_=self.lstms[lstm_layer](input_,self.lstms_state[lstm_layer])	


      input_ = tf.reshape(input_,[self.batch_size,-1])
      input_ = tf.matmul(input_,self.STN_W[0])+self.STN_b[0]
      input_ = tf.maximum(0.2 * input_, input_)
      input_ = tf.concat([input_,self.ratio],axis=1)
      input_ = tf.matmul(input_,self.STN_W[1])+self.STN_b[1]
      	
      scope.reuse_variables()	

      def forward():
        output_list = []
	cell_state = []
	param_list = []
	lstm_output_list = []
	
        for layer_lstm in range(self.layer_num_lstm):
          cell_state.append([])
	
	feature_list = []
	input_ = self.input_features[:,:,:,:]	
	#input_ = feature_extract(self.input_feature[:,:,:,:])
 	#input_ =self.input_features[:,i,:,:,:]
	for lstm_layer in range(self.layer_num_lstm):
          input_,self.lstms_state[lstm_layer]=self.lstms[lstm_layer](input_,self.lstms_state[lstm_layer])
	  cell_state[lstm_layer]= self.lstms_state[lstm_layer]

        lstm_output_list.append(input_)
	  	  
	input_ = tf.reshape(input_,[self.batch_size,-1])
        input_ = tf.matmul(input_,self.STN_W[0])+self.STN_b[0]
        input_ = tf.maximum(0.2 * input_, input_)
	input_ = tf.concat([input_,self.ratio],axis=1)
	input_ = tf.matmul(input_,self.STN_W[1])+self.STN_b[1]

	params = tf.stack([tf.multiply(input_[:,0],1.0/self.ratio[:,0]),tf.zeros(shape=self.batch_size),input_[:,1],tf.zeros(shape=self.batch_size),input_[:,0],input_[:,2]],axis=1)
	output= transformer(self.transform_matrix,params,[224,224])
	#return tf.stack(output_list,axis=1), tf.stack(param_list,axis=1),tf.stack(lstm_output_list,axis=1)
	return output, tf.stack(cell_state),input_[:,0],input_[:,1::]

      self.logits,self.cell_state,self.crop_size,self.center = forward() #note: output is logits need to convert to sigmoid in testing mode
  	

