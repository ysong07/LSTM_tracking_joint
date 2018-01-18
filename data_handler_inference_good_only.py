import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
#from params import *
import cv2
import time
import cProfile

with open('./badimages_id.txt','r') as file :
  lines = []
  temp = file.readlines()
  file_dict = {}
  for item in temp:
    nobad = True
    bad_image_list = []

    parsed_line = re.split('\t',item)

    for sub_item in parsed_line:
      if sub_item=='none':
        nobad = False
        break
      if re.search('-',sub_item) is not None:
        bad_id_list = re.split(',',re.sub('\n','',sub_item))
        for term in bad_id_list:
          trial_list = re.split('-',term)
          if len(trial_list)==2:
            min_val = int(trial_list[0][1::])
            max_val = int(trial_list[1][0:-1])
            for bad_id in range(min_val,max_val+1):
              bad_image_list.append(bad_id)

    file_name = parsed_line[0]
    file_name = re.sub(' ','',file_name)
    file_dict[file_name] = bad_image_list

    print 'file_name: ' +file_name +' bad list:'
    print  bad_image_list
  print 1

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
  
class data_handler_(object):
  def __init__(self,sess,length,train_file_name,batch_size=1):
    self.batch_size = batch_size
    self.length = length
    self.sess = sess
   
    self.data_file_ = h5py.File('../data_variable/'+re.sub('origin','',train_file_name),'r')
    self.just_image_file_ = h5py.File('../data_origin/'+train_file_name,'r')
    self.mask_file = h5py.File('../data_mask/'+ re.sub('origin','',train_file_name),'r')

	
    #self.data_file_ =h5py.File('../data/ball1.h5','r')
    self.height = self.data_file_['image'].shape[1]
    self.width = self.data_file_['image'].shape[2]
    #np.random.seed(100)
    
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - 32-1)

    bad_list = np.array(file_dict[re.sub('origin','',train_file_name)[0:-3]])   
    self.indices_ = np.setdiff1d(self.indices_,bad_list) 


    self.indices_id = 0	
    self.id_ = 0 # every hdf5 ids
    self.file_id = 0 # all file ids
    

  def Get_all(self):
    image = self.just_image_file_['image'][self.indices_[self.indices_id]+self.id_]
    image = np.stack([image],axis=0)
    mask = self.mask_file['image'][self.indices_[self.indices_id]+self.id_]
    mask = np.stack([mask],axis=0)
    return image, mask
	
  def Get_all_label(self):
    image = self.just_image_file_['image'][self.indices_[self.indices_id]+self.id_].astype('float')
    image = np.stack([image],axis=0)
    mask = self.mask_file['image'][self.indices_[self.indices_id]+self.id_].astype('float')
    mask = np.stack([mask],axis=0)
    label = self.just_image_file_['label'][self.indices_[self.indices_id]+self.id_]
    return image, mask,label


  def set_id(self,id):
    self.id_ = id   

  def set_indices_id(self):
    self.indices_id = self.indices_id+1 
    self.id_ = 0

  def set_file_id(self):
    self.data_file_.close()
    self.just_image_file_.close()
    self.mask_file.close()
    self.file_id +=1
    if self.file_id >= self.list_len :
      self.file_id = 0

    self.data_file_ = h5py.File('../data_variable/'+re.sub('origin','',self.list_files[self.file_id]),'r')
    self.just_image_file_ = h5py.File('../data_origin/'+self.list_files[self.file_id],'r')
    self.mask_file = h5py.File('../data_mask/'+ re.sub('origin','',self.list_files[self.file_id]),'r')

    self.height = self.data_file_['image'].shape[1]
    self.width = self.data_file_['image'].shape[2]

 
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - 32-1)

    bad_list = np.array(file_dict[re.sub('origin','',self.list_files[self.file_id])[0:-3]])
    self.indices_ = np.setdiff1d(self.indices_,bad_list)
	
    self.id_ = 0
    np.random.shuffle(self.indices_)

 
  def generate_group_image(self,images,predict,truth,file_name,id):
    for i in range(images.shape[0]):
      plt.subplot(2,1,1)
      plt.imshow(images[i,:,:,:]/255.0,alpha=0.9)
      plt.imshow(predict[i,:,:,:].squeeze(),alpha=0.3)
      plt.axis('off')
      plt.subplot(2,1,2)
      plt.imshow(images[i,:,:,:]/255.0,alpha=0.9)
      plt.imshow(cv2.cvtColor(np.uint8(truth[i,:,:,:].squeeze()),cv2.COLOR_BGR2GRAY)/255.0,alpha=0.3)
      plt.axis('off')
      plt.savefig(file_name+str(id+i),bbox_inches='tight')
      plt.clf()
 

  def generate_overlay_image(self,images,predict,file_name,id):
    for i in range(images.shape[0]):
      plt.imshow(images[i,:,:,:]/255.0,alpha=0.9)
      plt.imshow(predict[i,:,:,:].squeeze(),alpha=0.3)
      plt.axis('off')
      plt.savefig(file_name+str(id+i), bbox_inches='tight')
      plt.clf()

  

if __name__ =="__main__":
   sess = tf.Session()
   data_handler = data_handler_(sess,2,'/scratch/ys1297/LSTM_tracking/data_origin/train_list_0.txt')
   for i in range(5): 
     t = time.time() 
#     feature_0_mask,feature_0_no_mask, all_features, all_masks,all_images = data_handler.GetBatch()
     print time.time()-t
     #data_handler.generate_overlay_image(all_images[0,:,:,:,:],all_masks[0,:,:,:,:],'test'+str(i))
 
     print i



	
