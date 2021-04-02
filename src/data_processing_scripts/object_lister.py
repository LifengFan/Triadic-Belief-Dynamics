#!/usr/bin/env python
import sys
import time
import rospy
import cv2
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from tf import TransformBroadcaster, transformations
from cv_bridge import CvBridge, CvBridgeError
from threading import Thread, Lock
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import os
from sys import platform
import argparse
import time
from darknet import *

# VALUES YOU MIGHT WANT TO CHANGE
COLOR_IMAGE_TOPIC = "azure/color_image"  # Ros topic of the undistorted color image


class KinectDataSubscriber:
    """ Holds the most up to date """
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber(COLOR_IMAGE_TOPIC,
                                          Image,
                                          self.color_callback, queue_size=1, buff_size=10000000)                                
        # data containers and its mutexes
        self.color_image = None
        self.color_mutex = Lock()


    def color_callback(self, data):
        """ Called whenever color data is available. """
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.color_mutex.acquire()
        self.color_image = cv_image
        self.color_mutex.release()

net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
meta = load_meta("cfg/coco.data")

""" Main """
if __name__ == '__main__':
    # Start Node that reads the kinect topic
    image_data = KinectDataSubscriber()
    rospy.init_node('listener', anonymous=True) 

    # loop
    try:
        while not rospy.is_shutdown():
            image_data.color_mutex.acquire()
        
            if (image_data.color_image is not None):
                color = image_data.color_image.copy()
                image_data.color_mutex.release()
                draw_img = color.copy()
                r = detect_np(net, meta, color)
                print(r)
                for object_ in r:
                    cv2.rectangle(draw_img, (int(object_[2][0] - object_[2][2]/2), int(object_[2][1] - object_[2][3]/2)), 
                            (int(object_[2][0] + object_[2][2]/2), int(object_[2][1] + object_[2][3]/2)), (255,0,0), 2)
                    cv2.putText(draw_img, object_[0], (int(object_[2][0]-10), int(object_[2][1] - 10)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,0), 1, cv2.LINE_AA)
                cv2.imshow('window', draw_img)
                cv2.waitKey(100)
                

            else:
                image_data.color_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
