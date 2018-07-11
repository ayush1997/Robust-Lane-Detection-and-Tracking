#!/usr/bin/env python
# from __future__ import print_function
# import torch

# import tensorflow as tf
import numpy as np
import time
import roslib
import math
roslib.load_manifest('swarath_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from swarath_package.msg import lstm_data
from skimage import io
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import os
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np 
import cv2
import math
import pickle
from std_msgs.msg import String


f_r = KalmanFilter (dim_x=4, dim_z=4)
# f = ExtendedKalmanFilter (dim_x=2, dim_z=1)

f_l = KalmanFilter (dim_x=4, dim_z=4)


f_r.x = np.array([[285],[80],[587],[260]])   # velocity
# f.x = np.array([[350],[100],[55],[10]])   # velocity
f_l.x = np.array([[331],[80],[156],[265]])   # velocity

# print (f_r.x.shape)
# f.F= np.array([[0.8]])   
# f.F = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])

# state transistion matrix

f_r.F = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

f_l.F = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])


# measurement function

f_r.H=np.array([[1.,0.,0,0],
    [0.,1.,0,0],
    [0,0,1,0],
    [0,0,0,1]])


f_l.H=np.array([[1.,0.,0,0],
    [0.,1.,0,0],
    [0,0,1,0],
    [0,0,0,1]])



# covariance matrix

f_r.P = np.array([[100,0,0,0],
    [ 0, 100,0,0],
    [0,0,100,0],
    [0,0,0,100] ])

f_l.P = np.array([[100,0,0,0],
    [ 0, 100,0,0],
    [0,0,100,0],
    [0,0,0,100] ])


# measurement uncertainty/noise

# f_r.R = np.array([[100,0.,0,0],
#     [ 0.,200,0,0],
#     [0,0,300,0],
#     [0,0,0,400] ])

# f_l.R = np.array([[100,0.,0,0],
#     [ 0.,200,0,0],
#     [0,0,300,0],
#     [0,0,0,400] ])

f_r.R = np.array([[100,0.,0,0],
    [ 0.,100,0,0],
    [0,0,100,0],
    [0,0,0,100] ])

f_l.R = np.array([[100,0.,0,0],
    [ 0.,100,0,0],
    [0,0,100,0],
    [0,0,0,100] ])

lane_not_detect_count_left = 0
lane_not_detect_count_right = 0





class cnn_lstm(torch.nn.Module):
    def __init__(self,feature,hidden_unit, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(cnn_lstm, self).__init__()
        self.model_ft = models.alexnet(pretrained=True)
        # print (model_ft)

        self.num_ftrs = self.model_ft.classifier[6].in_features
        self.feature_model = list(self.model_ft.classifier.children())
        self.feature_model.pop()
        self.feature_model.pop()
        # feature_model.append(nn.Linear(num_ftrs, 3))
        self.feature_model.append(nn.Linear(self.num_ftrs, 1046))
        self.feature_model.append(nn.Linear(1046, 100))

        self.model_ft.classifier = nn.Sequential(*self.feature_model)

        self.rnn = nn.LSTM(feature,hidden_unit,batch_first=True).cuda()
        self.linear = torch.nn.Linear(D_in, D_out).cuda()


    def forward(self,x):
        
        fc1 = self.model_ft(x)
        fc1 = torch.unsqueeze(fc1,2)
        # print (fc1.size())

        rnn,(_,_) = self.rnn(fc1)
        # print (rnn)
        # print (rnn[:,-1])
        y_pred = self.linear(rnn[:,-1])
        # print (y_pred)
        return y_pred

def process_point(pt1,pt2,xmax,ymax):

  # print (xmax,ymax)
  m = float(pt2[1]-pt1[1])/(pt2[0]-pt1[0])
  c = float(pt1[1] - m*pt1[0])

  # if m >0 :
  #   c = -c

  xmin = ymin = 0

  # x =0
  x1 = 0 
  y1 =c  
  # x=xmax 
  x2 = xmax 
  y2 = m*x2 + c
  # y = 0
  x3 = -c/m
  y3 = 0 
  # y = ymax 
  x4 = (ymax-c)/m
  y4 = ymax 

  # print x1,y1
  # print x2,y2
  # print x3,y3
  # print x4,y4
  point = []
  for x,y in zip([x1,x2,x3,x4],[y1,y2,y3,y4]):
    if (x>=0 and x<=xmax) and (y>=0 and y<=ymax):
      point.append([int(x),int(y)])

  # print "UP",unique_point
  # print "P",point
  # point = unique_point
  if len(point) !=2:
    unique_point = [list(x) for x in set(tuple(x) for x in point)]
    if len(unique_point) == 2:
      return unique_point
    else:
      print ("erro==================")


  else:
    return point
    




trans = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
])

def process_points(points):
  filt = 4
  y = len(points)/filt
  final_points = []

  for x in range(y):
    final_points.append(list(points[x*filt:x*filt + filt]))

  # print (final_points)
  return final_points




alexnet_model = torch.load("/home/ayush/Documents/swarath_lane/swarath/src/lanedetection_shubhamIITK/src/exp_aug.pkl")
# print alexnet_model
count = 0

right_lane_cov = []
left_lane_cov = []

def prep_data(org_image,cv_image,f_points):
  global count
  global lane_not_detect_count_left
  global lane_not_detect_count_right
  global right_lane_cov
  global left_lane_cov

  # msg_cov = []
  msg_cov = ""

  # pub = rospy.Publisher('/chatter', String)
  # rospy.init_node('prep_data', anonymous=True)


  # rate = rospy.Rate(8)

  img  = cv_image

  ymax,xmax = img.shape[:2]

  # cv2.imshow("Image   window", cv_image)
  # cv2.waitKey(3)
  org = img.copy()


      

  point1 = []
  point2 = []

  for i in f_points:
      point1.append([i[0],i[1]])
      point2.append([i[2],i[3]])


  # print (point1,point2)


  
  # count = 0
  start_points = []
  end_points = []
  lane_type = []
  image_vec = []

  for i,j in zip(point1,point2):
    
    i,j = process_point(i,j,xmax,ymax)

    start_points.append(i)
    end_points.append(j)

    offset = 15
    # offset = 25
    a = (i[0]-offset,i[1])
    b = (i[0]+offset,i[1])
    c = (j[0]+offset,j[1])
    d = (j[0]-offset,j[1])

    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[a,b,c,d]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your img
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_img = cv2.bitwise_and(img, mask)
    # print masked_img.shape
    # plt.imshow(masked_img)
    # plt.show()
    #for curved lane
    # masked_img_cop = masked_img[60:,:,:]
    # plt.imshow(masked_img)
    # plt.show()

    # print i,j

    k1 = 70
    k2 = 30
    z = [0,0]
    if i[1] < j[1] :
      
      z[0] = (k1*i[0] + k2*j[0])/(k1+k2)
      z[1] = (k1*i[1] + k2*j[1])/(k1+k2)

      if i[0]<j[0]:
        # crop_img = masked_img[i[1]:j[1], i[0]:j[0]]
        crop_img = masked_img[z[1]:j[1], z[0]:j[0]]
      elif i[0]>j[0]:
        # crop_img = masked_img[i[1]:j[1], j[0]:i[0]]
        crop_img = masked_img[z[1]:j[1], j[0]:z[0]]
      elif i[0] == j[0]:
        # crop_img = masked_img[i[1]:j[1], j[0]-offset:j[0]+offset]
        crop_img = masked_img[z[1]:j[1], j[0]-offset:j[0]+offset]
    elif i[1] > j[1]:
      z[0] = (k1*j[0] + k2*i[0])/(k1+k2)
      z[1] = (k1*j[1] + k2*i[1])/(k1+k2)


      if i[0]<j[0]:
        # crop_img = masked_img[j[1]:i[1], i[0]:j[0]]
        crop_img = masked_img[z[1]:i[1], i[0]:z[0]]
      elif i[0] > j[0]: #right lane 
        # crop_img = masked_img[j[1]:i[1], j[0]:i[0]]
        crop_img = masked_img[z[1]:i[1], z[0]:i[0]]
      elif i[0] == j[0]:
        # crop_img = masked_img[j[1]:i[1], j[0]-offset:j[0]+offset]
        crop_img = masked_img[z[1]:i[1], j[0]-offset:j[0]+offset]
        # crop_img = masked_img[z[1]:i[1], cdj[0]-offset:j[0]+offset]
    # elif i[0] == j[0]:
    #   print "true"
    #   if i[1]<j[1]:
    #     crop_img = masked_img[i[1]:j[1], i[0]-offset:i[0]+offset]
    #   else:
    #     crop_img = masked_img[j[1]:i[1], j[0]-offset:j[0]+offset]


    # print masked_img_cop.shape

    # plt.imshow((cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)))
    # cv2.imwrite("results/curve_data/"+str(n)+".jpg",masked_img)

    # plt.show()

    crop_img = cv2.resize(crop_img,(224,224))

    # print "crop img",crop_img.shape

    pil_im = Image.fromarray(crop_img).convert('RGB')
    inputs =  trans(pil_im).view(3, 224, 224)
    # print inputs
    # print (pil_im)
    image_vec.append(inputs)

    # plt.imshow((cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)))
    # cv2.imwrite("result_aug/"+str(n)+".jpg",crop_img)

    # cv2_im = cv2.cv
    # tColor(crop_img,cv2.COLOR_BGR2RGB)


  if len(image_vec)!=0:
    inputs = torch.stack(image_vec)
    
    # pil_im = Image.fromarray(crop_img).convert('RGB')
    # inputs =  trans(pil_im).view(1, 3, 224, 224)

    inputs = Variable(inputs.cuda())

    # print (inputs)
    outputs = alexnet_model(inputs)
    o = outputs.data.cpu().numpy()
    # print (o)
    preds =  np.argmax(o,axis=1)
    # print ("pred",preds)

    # e_x = np.exp(o[0] - np.max(o[0]))
    # score = e_x / e_x.sum(axis=0)
    # print "pred",(pred)
    # # print "score",(score)

    # sky = 260
    sky = 185


    f_r.predict()
    f_l.predict()


    pred_r = f_r.x
    pred_l = f_l.x

 
    pred_r_cov = f_r.P
    pred_l_cov = f_l.P

    left_lane_cov.append(np.diagonal(pred_l))
    right_lane_cov.append(np.diagonal(pred_r))

    hello_str = ""


    # if list(np.diagonal(pred_l_cov)) <= [29.0,34.5,37.0,41.0]:
    # if list(np.diagonal(pred_l_cov)) <= [29.0,29.0,29.0,29.0]:

    #   # hello_str = str([i[0] for i  in pred_l])
    #   for i  in pred_l:
    #     hello_str = hello_str + str(i[0]) + " " 

    #   # msg_cov.append(hello_str)
    #   msg_cov += hello_str

      
    # else:
    #   # msg_cov.append("False")
    #   msg_cov += "False "
      
      
    # hello_str = ""
    
    # # if list(np.diagonal(pred_r_cov)) <= [57.0,60.5,64.0,67.0]:
    # if list(np.diagonal(pred_r_cov)) <= [56.0,56.0,56.0,56.0]:
    #   # msg_cov.append("True")
    #   # hello_str = str([i[0] for i  in pred_r])

    #   for i  in pred_l:
    #     hello_str = hello_str + str(i[0]) + " " 

    #   msg_cov += hello_str


    #   # msg_cov.append(hello_str)

      
    # else:
    #   # msg_cov.append("False")
    #   msg_cov += "False "
      

    # pub.publish(str(msg_cov))  
    # rate.sleep()


    # print pred_r
    # print pred_l
    # print left_lane_cov
    # print right_lane_cov

    l_slope = []
    r_slope = []

    for i,j,pred in zip(point1,point2,preds):

      i[1]+=sky
      j[1]+=sky

      x_scale = 964/480.0
      y_scale = 1288/640.0

      i[0] = int(i[0]*x_scale)
      i[1] = int(i[1]*y_scale)
      j[0] = int(j[0]*x_scale)
      j[1] = int(j[1]*y_scale)

      
   
    # for line in file:
    
      



      # if pred == 0 and np.max(score)>0.85 :
      if pred == 0  :

        # print "before process",i,j
        i,j = process_point(i,j,1288,964)

        # print "processed",i,j

        k1 = 40
        k2 = 60
        z = [0,0]
        if j[1] < i[1] :

          j[0] = (k1*j[0] + k2*i[0])/(k1+k2)
          j[1] = (k1*j[1] + k2*i[1])/(k1+k2)


        else:
          i[0] = (k1*i[0] + k2*j[0])/(k1+k2)
          i[1] = (k1*i[1] + k2*j[1])/(k1+k2)


        if i[0]-j[0] !=0:
            slope = (i[1]-j[1])/float((i[0]-j[0]))
            m = np.degrees(np.arctan(math.fabs(slope)))
        else:
          slope = 1
          m = 90
          
        # print (m)
        # print (slope)
        # if slope > 0 :
        if slope > 0 and (i[0]<=1288 and j[0]<=1288):
          r_slope.append((slope,i,j))
        # elif slope < 0 :
        elif slope < 0 and (i[0]>=0 and j[0]>=0):
          l_slope.append((slope,i,j))


        # cv2.line(org,tuple(i),tuple(j),color=(0,255,0),thickness=2)

        # cv2.line(org_image,tuple(i)+tuple(),tuple(j),color=(0,255,0),thickness=2)

      # print "score",(score)

      # if pred == 0 and np.max(score)>0.85 :
      # # if pred == 0  :
      #   cv2.line(org,tuple(i),tuple(j),color=(0,255,0),thickness=2)
      # else:
      #   cv2.line(img,tuple(i),tuple(j),color=(0,0,255))
          
      # plt.imshow((cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
      # plt.imshow((cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)))
      # plt.show()
    # SaveFigureAsImage("/home/ayush/Documents/swarath_confidence_new/model_test/"+str(n), plt.gcf() , orig_size=(h,w))

      # cv2.imshow("Image window", org_image)
      # cv2.waitKey(3)

  ###########################################3

    r_slope.sort(key=lambda x: x[0],reverse=True)
    l_slope.sort(key=lambda x: x[0],reverse=False)

    if len(r_slope)!=0:
      if ([r_slope[0][1][1] > r_slope[0][2][1]]):
        f_r.update(np.array([[r_slope[0][1][0]],[r_slope[0][1][1]],[r_slope[0][2][0]],[r_slope[0][2][1]]]))
        lane_not_detect_count_right = 0
      else:
        f_r.update(np.array([[r_slope[0][2][0]],[r_slope[0][2][1]],[r_slope[0][1][0]],[r_slope[0][1][1]]]))
        lane_not_detect_count_right = 0
    else:
      # f_r.update(f_r.x)
      lane_not_detect_count_right+=1


    if len(l_slope)!=0:
      if ([l_slope[0][1][1] > l_slope[0][2][1]]):
        f_l.update(np.array([[l_slope[0][1][0]],[l_slope[0][1][1]],[l_slope[0][2][0]],[l_slope[0][2][1]]]))
        lane_not_detect_count_left = 0
      else:
        f_l.update(np.array([[l_slope[0][2][0]],[l_slope[0][2][1]],[l_slope[0][1][0]],[l_slope[0][1][1]]]))
        lane_not_detect_count_left = 0
        
    else:
      # f_l.update(f_l.x)
      lane_not_detect_count_left += 1



    # print "r_slope",r_slope
    # print "l_slope",l_slope

    if len(r_slope) != 0 :  
      cv2.line(org_image,tuple([r_slope[0][1][0],r_slope[0][1][1]]),tuple([r_slope[0][2][0],r_slope[0][2][1]]),color=(0,0,255),thickness=2)
    if len(l_slope) !=0:
      cv2.line(org_image,tuple([l_slope[0][1][0],l_slope[0][1][1]]),tuple([l_slope[0][2][0],l_slope[0][2][1]]),color=(0,0,255),thickness=2)

    if lane_not_detect_count_left < 5: 
      cv2.line(org_image,tuple([int(pred_l[0][0]),int(pred_l[1][0])]),tuple([int(pred_l[2][0]),int(pred_l[3][0])]),color=(0,255,0),thickness=2)
    if lane_not_detect_count_right < 5: 
      cv2.line(org_image,tuple([int(pred_r[0][0]),int(pred_r[1][0])]),tuple([int(pred_r[2][0]),int(pred_r[3][0])]),color=(0,255,0),thickness=2)
    



    # out.write(org_image)
    cv2.imshow('Frame',org_image)
    cv2.waitKey(3)


    # cv2.imwrite("./result/dnd_kal/"+str(file_name)+".jpg",img)
    # cv2.imwrite("./result/india_gate_kal/"+str(file_name)+".jpg",img)
    # cv2.imwrite("./result/india_kal/"+str(file_name)+".jpg",img)

    # Press Q on keyboard to  exit
    # if cv2.waitKey(50) & 0xFF == ord('q'):
    #   break


  ################################3

    cv2.imwrite("/home/ayush/Documents/swarath_lane/swarath/src/lanedetection_shubhamIITK/src/kalman_image_test/"+str(count)+".jpg", org_image)
      # val_file.write("%s\n" %(str(n)+"_"+str(count)+".jpg")) 
      
    count+=1
    # plt.show()

    # Gradients

  else:
    cv2.imshow('Frame',org_image)
    cv2.waitKey(3)
    cv2.imwrite("/home/ayush/Documents/swarath_lane/swarath/src/lanedetection_shubhamIITK/src/kalman_image_test/"+str(count)+".jpg", org_image)

    count+=1

        

class image_converter:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/lstm",lstm_data,self.callback)
    self.img_no = 0

  def callback(self,data):

    print ("called")
    start_time = time.time()
    
    try:
      # rospy.loginfo("%s is age: %d" % (data.im))
      # print (data.label)
      # print (data.points)
      cv_image = self.bridge.imgmsg_to_cv2(data.im, "bgr8")

      org_image = cv_image.copy()

      # print ("org image",org_image.shape)

      cv_image =  cv2.resize(cv_image,(640,480))

      bonnet = 430
      # sky = 260
      sky = 185

      cv_image = cv_image[sky:bonnet,0:640]

      # print (cv_image.shape)




      # cv_image_org = self.bridge.imgmsg_to_cv2(data.im_org, "bgr8")

      # print (data.points)

      # if data.label[0] > 0:

      f_points = process_points(data.points)
      # lane_type = process_label(data.label)

      # print (len(lane_type))
      # print (len(f_points))

      # print (f_points)

      prep_data(org_image,cv_image,f_points)
        


    except CvBridgeError as e:
      print(e)

   

    # cv2.imshow("Image window", cv_image)

    # cv2.imwrite(path + str(self.img_no)+".jpg",cv_image);
    print("saved")
    self.img_no +=1

    print ("time take --> ",time.time()-start_time)
    # cv2.waitKey(3)

    

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)



  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':


    main(sys.argv)


    # FNR = "righ_cov"
    # FNL = "left_cov"
    # # open the file for writing
    # fileObject_r = open(FNR,'wb') 
    # fileObject_l = open(FNL,'wb') 
    # pickle.dump(left_lane_cov,fileObject_r)   
    # pickle.dump(right_lane_cov,fileObject_l)   


