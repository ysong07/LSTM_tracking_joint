import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
import BasicConvLSTMCell
#from VGG import *

data_dict =np.load('../model/params.npy').item()
#data_dict = np.load('../model/params_fcn.npy').item()
def feature_extract(batch_frames,mask_frames):
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
    conv1_2 = tf.multiply(conv1_2,mask_frames)
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



class G_model_:
  def __init__ (self,scope,height,width,batch_size,layer_num_lstm,kernel_size,kernel_num,kernel_size_dec,num_dec_input,num_dec_output,layer_num_cnn,initial_h_0,initial_c_0,initial_h_1,initial_c_1,img_height,img_width):
    self.scope = scope
    self.height = height
    self.width = width
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


    self.img_height = img_height
    self.img_width = img_width 
    self.define_graph()

  def define_graph(self):
    with tf.name_scope('input'):
      #self.first_mask = tf.placeholder(tf.float32,shape=None)
      self.input_frame = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_height, self.img_width,3])
      self.center = tf.placeholder(tf.float32,shape=[2])
      self.crop_size = tf.placeholder(tf.float32, shape = [1])
      self.input_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_height, self.img_width,1])
 
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
      
      self.dec_conv_W = []
      self.dec_conv_b = []	
      for layer_id_,kernel_,input_,output_ in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
        with tf.variable_scope("dec_conv_{}".format(layer_id_)):
          self.dec_conv_W.append(tf.get_variable("matrix", shape = kernel_+[output_,input_], initializer = tf.random_uniform_initializer(-0.01, 0.01)))
          self.dec_conv_b.append(tf.get_variable("bias",shape = [output_],initializer=tf.constant_initializer(0.01)))
	  
      input_ = tf.zeros((self.batch_size,14,14,512))
      #input_ = tf.zeros((self.batch_size,28,28,256))
      for layer_lstm in range(self.layer_num_lstm):
        input_,_=self.lstms[layer_lstm](input_,self.lstms_state[layer_lstm])	

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
      
      def get_template(frame,center,crop_size):
	# input: 
 	#   search frame
	#   center
        # output:
	#   template (size 448 *448)

	pad_frame = tf.image.pad_to_bounding_box(frame,crop_size/2,crop_size/2,self.img_height+crop_size,self.img_width+crop_size)
	output = tf.image.crop_to_bounding_box(pad_frame,center[0],center[1],crop_size,crop_size)

        return output

       
	
      def get_center(mask):
	# get image mask center
	yv,xv = tf.meshgrid(tf.range(start=0,limit=self.img_width),tf.range(start=0,limit=self.img_height))
	mask = tf.squeeze(mask)
        xv = tf.to_float(xv)
        yv = tf.to_float(yv)
	if tf.reduce_sum(mask)==0:
	  return None
	#center_x,center_y = tf.to_int32(tf.reduce_sum(xv*mask)/tf.reduce_sum(mask)),tf.to_int32(tf.reduce_sum(yv*mask)/tf.reduce_sum(mask))
	center_x,center_y = tf.reduce_sum(xv*mask)/tf.reduce_sum(mask),tf.reduce_sum(yv*mask)/tf.reduce_sum(mask)
	#center_x = tf.cond(center_x<=0,lambda:tf.constant([0]),lambda:center_x)
	#center_y = tf.cond(center_y<=0,lambda:tf.constant([0]),lambda:center_y)
        #center_x = tf.cond(center_x>=self.img_height-1, lambda:self.img_height-1,lambda:center_x)
	#center_y = tf.cond(center_y>=self.img_width-1, lambda:self.img_width-1,lambda:center_y)
	
	center = [center_x,center_y]	
        return center

      def fill_template(mask,center,crop_size):
	# mask : template prediction
	# crop_center: cropped location
        try:	
	  temp_img = tf.image.pad_to_bounding_box(mask,center[0],center[1] ,self.img_height+crop_size,self.img_width+crop_size)
          return tf.image.crop_to_bounding_box(temp_img,crop_size/2,crop_size/2,self.img_height,self.img_width)
	except: 
	  return None
      def get_radius(mask):
	return tf.round(tf.sqrt(tf.reduce_sum(mask)))

      def forward():
        output_list = []
	center_list = []
        cell_state = []
	output_template_list = []
	predict_template_list = []
 	
	lstm_output = []	
	for layer_lstm in range(self.layer_num_lstm):
	  cell_state.append([])
	
        center = [self.center[0],self.center[1]]
	ema_center = [tf.to_float(center[0]),tf.to_float(center[1])]
	center = [tf.to_int32(center[0]),tf.to_int32(center[1])] 

	crop_size = self.crop_size[0]
	crop_size = tf.minimum(crop_size,224*3)
	crop_size = tf.maximum(crop_size,56) 	    
        crop_size = tf.to_int32(crop_size)
	
	template = get_template(self.input_frame[0,:,:,:],center,crop_size)
	template = tf.image.resize_images(template,tf.constant([224,224]),method=0)
	template = tf.expand_dims(template,axis=0) 
	
	mask_template = get_template(self.input_mask[0,:,:,:],center,crop_size)
	mask_template = tf.image.resize_images(mask_template,tf.constant([224,224]),method=0)
    
        # extract feature
        input_ = feature_extract(template,tf.ones([self.batch_size,224,224,64]))
        feature_output = input_
	# convolutional LSTM
	for layer_lstm in range(self.layer_num_lstm):
          input_,self.lstms_state[layer_lstm]=self.lstms[layer_lstm](input_,self.lstms_state[layer_lstm])	
          cell_state[layer_lstm].append(self.lstms_state[layer_lstm])
	lstm_output.append(input_)
        # deconvolution 	 
	for layer_id_,kernel_num_,num_input,num_output in zip(xrange(self.layer_num_cnn),self.kernel_size_dec,self.num_dec_input,self.num_dec_output):
          if layer_id_ == self.layer_num_cnn-1:
            #output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
	    output_shape = tf.constant([1,224,224,1])
            input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
	    output_template = tf.sigmoid(input_)
	    output_logits = input_
          else:
            output_shape = tf.stack([tf.shape(input_)[0],tf.shape(input_)[1]*2,tf.shape(input_)[2]*2,num_output])
            input_ = tf.nn.conv2d_transpose(input_, self.dec_conv_W[layer_id_],output_shape = output_shape, strides= [1,2,2,1],padding= 'SAME') + self.dec_conv_b[layer_id_]
            input_ = tf.maximum(0.2 * input_, input_)

        output_template = tf.squeeze(output_template,axis=0)
	output_template = tf.image.resize_images(output_template,[crop_size,crop_size],method=0)
	output_frame = fill_template(output_template,center,crop_size)
        return_crop_size = crop_size 	
        return output_frame,output_template, tf.stack(cell_state),template, mask_template,feature_output,return_crop_size,tf.stack(lstm_output,axis=1) 

      self.predicts,self.templates,self.cell_states,self.original_template,self.mask_template,self.feature_list,self.return_crop_size,self.lstm_output =  forward()


