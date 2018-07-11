#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import roslib
roslib.load_manifest('swarath_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from swarath_package.msg import lstm_data

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('/home/ayush/Documents/swarath_lane/swarath/src/lanedetection_shubhamIITK/src/model/my_test_model_480.meta')
# saver.restore(sess,tf.train.latest_checkpoint('/home/ayush/Documents/swarath_lane/swarath/src/lanedetection_shubhamIITK/src/model/'))
saver.restore(sess,tf.train.latest_checkpoint('/home/ayush/Documents/swarath_lane/swarath/src/lanedetection_shubhamIITK/src/new_model/'))

graph = tf.get_default_graph()

pred = graph.get_tensor_by_name("pred:0")

data = graph.get_tensor_by_name("input:0")

print ("loaded")




def pred_and_plot(final_label):

  predictions = sess.run(pred,{data: final_label})

  lane_type = []
  for pd in predictions:
    if np.argmax(pd) == 0:
     lane_type.append("dotted")
    elif np.argmax(pd) == 1:
      lane_type.append("false")
    elif np.argmax(pd) == 2:
      lane_type.append("solid")

  return lane_type

def make_feature_list(a):
  final_label = []
  for i in range(len(a)):
    filt = 10
    y = len(a[i])/filt

    # print np.array(final_points[i])[:,0]
    global_count = []
    for x in range(10):
      g_label=0
      r_label=0

      for label in a[i][x*y:x*y + y]:
        if label == 1:
          g_label += 1
        else:
          r_label += 1

      global_count.append([g_label])
      global_count.append([r_label])

    # print global_count
    # print len(global_count)

    final_label.append(global_count)

  for i in final_label:
    print (i)

  lane_type = pred_and_plot(final_label)
  return lane_type
  




def process_label(label):
  label = list(label)
  lines = label[0]
  label =  label[1:]
#   print label

  global_list = []
  for l in range(lines):
    length = label[0]
    # print length
    label_ = label[1:length+1]
    print (label_)
    print (len(label[1:length+1]))

    label = label[length+1:]
    print (label)
    global_list.append(label_)

  print (global_list)
  lane_type = make_feature_list(global_list)
  return lane_type

def process_points(points):
  filt = 4
  y = len(points)/filt
  final_points = []

  for x in range(y):
    final_points.append(list(points[x*filt:x*filt + filt]))

  print (final_points)
  return final_points





label =(2, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)




# f_points = process_points(points)
lane_type = process_label(label)

  