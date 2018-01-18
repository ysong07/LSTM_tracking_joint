import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
import os
import time
import sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import sys 
def calculate_IOU(GT, predict,thre=0.2):
    predict = np.reshape(predict,(1,-1))
    GT = np.reshape(GT,(1,-1))
    label1  = np.where(predict>thre)[1]
    label2 = np.where(GT ==255.0)[1]
    intersect = np.intersect1d(label1,label2)
    union = np.union1d(label1,label2)
   
    IOU = float(intersect.size)/(union.size+0.000001)
    if intersect.size == 0:
      no_inter_flag = False
    else:
      no_inter_flag = True 
    return IOU, no_inter_flag


if __name__ =='__main__':
    file_name = sys.argv[1]
    CODE_TYPE = cv2.cv.CV_FOURCC('m','p','4','v')
    
    file= scipy_io.loadmat('./img/'+file_name+'/'+'0.mat')
    height,width = file['image'][0].shape[0],file['image'][0].shape[1]
#    video = cv2.VideoWriter('./img/'+ file_name+'.avi',CODE_TYPE,1,(height,width),1)
    
    for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
      IOU_list = []
	
      for i in range(1000):
        print i
        try:
          file= scipy_io.loadmat('./img/'+file_name+'/'+'{}.mat'.format(i))
          mask = file['mask'].squeeze()[:,:,0]
          predict = file['predict'].squeeze()
          IOU,flag=  calculate_IOU(mask,predict,threshold)
        except:
   	  break
        if flag :
  	 IOU_list.append(IOU)
  	# """ generating video """
          # plt.subplot(2,2,1)
          # plt.imshow(file['image'][0]/255.0,alpha=0.9)
          # plt.imshow(cv2.cvtColor(np.uint8(file['mask'][0]),cv2.COLOR_BGR2GRAY)/255.0,alpha=0.5,cmap ='hot')
          # #plt.imshow(np.uint8(file['mask'][0][:,:,0].squeeze()),alpha=0.3,cmap= 'hot')
          # plt.axis('off')
          # plt.title("image overlay with ground truth", y=1.05, fontsize=10)
  
          # plt.subplot(2,2,2)
          # plt.imshow(file['image'][0]/255.0,alpha=0.9)
          # plt.imshow(file['predict'].squeeze(),alpha=0.5,cmap= 'hot')
          # #plt.imshow(file['predict'][0].squeeze()/255.0,alpha=0.3)
          # #plt.imshow(np.uint8(file['mask'][0]).squeeze())
          # plt.axis('off')
          # plt.title("image overlay with prediction", y=1.05, fontsize=10)
  
          # plt.subplot(2,2,3)
          # plt.imshow(cv2.resize(file['glob_frame'][0],(width,height)).squeeze())
          # plt.axis('off')
          # plt.title("global model prediction", y=1.05, fontsize=10)
  
          # plt.subplot(2,2,4)
          # plt.imshow(np.uint8(file['g_templates'][0][:,:,::-1]))
          # #plt.imshow(np.uint8(file['g_templates'][:,:,::-1]))
          # #pdb.set_trace()
          # plt.imshow(cv2.resize(file['g_template_predict'].squeeze(),(224,224)),alpha=0.5,cmap ='hot')
          # plt.axis('off')
          # plt.title("local model prediction", y=1.05, fontsize=10)
  	# plt.savefig('./img/'+file_name+'{}.pdf'.format(i))
  #         video.write(img)
  
        else:
  #	 break
  	  IOU_list.append(0)
      with open('./img/'+file_name+'.txt','a') as txt_file:
	IOU_list = np.asarray(IOU_list).squeeze()
	txt_file.write(str(np.mean(IOU_list[0:20]))+','+str(np.mean(IOU_list[0:40]))+','+str(np.mean(IOU_list[0:80]))+','+str(np.mean(IOU_list[0:160]))+'\n')
 
#    print np.mean(np.asarray(IOU_list))
#    print len(IOU_list)
#    video.release()

    
     	
