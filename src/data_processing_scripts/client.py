#!/usr/bin/env python

import socket
from struct import *
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from tf import TransformBroadcaster, transformations
import numpy as np
import math

class CoordinatePublisher:
    def __init__(self):
        person1pub = rospy.Publisher('azure_pose1', MarkerArray, queue_size=1)
        person2pub = rospy.Publisher('azure_pose2', MarkerArray, queue_size=1)
        handpub1 = rospy.Publisher('hand_pose1', PointCloud2, queue_size=1)
        handpub2 = rospy.Publisher('hand_pose2', PointCloud2, queue_size=1)
        eyepub1 = rospy.Publisher('eye_pose1', PointCloud2, queue_size=1)
        eyepub2 = rospy.Publisher('eye_pose2', PointCloud2, queue_size=1)
        self.personspub = [person1pub, person2pub]
        self.handspub = [handpub1, handpub2]
        self.eyespub = [eyepub1, eyepub2]
        rospy.init_node('talker', anonymous=True)
        HOST = '192.168.217.1'    # The remote host
        PORT = 8080              # The same port as used by the server
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))
        self.s.sendall('Hello, world')
        self.coordi_num = 2*3*26+3
        self.br = TransformBroadcaster()
        self.camera_frame_id = 'azure/color_image'
        self.magnitude = 0.05

    def xyz_array_to_pointcloud2(self, points, stamp=None, frame_id=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        msg = PointCloud2()
        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12*points.shape[0]
        msg.is_dense = int(np.isfinite(points).all())
        msg.data = np.asarray(points, np.float32).tostring()

        return msg 

    def xyzrgb_array_to_pointcloud2(self, points, colors, stamp=None, frame_id=None, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        msg = PointCloud2()
        assert(points.shape == colors.shape)

        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if seq: 
            msg.header.seq = seq
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            N = len(points)
            print(N)
            xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
            msg.height = 1
            msg.width = N
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * N
        msg.is_dense = True
        msg.data = xyzrgb.tostring()

        return msg 

    def rotate_y(self, theta):
        return np.array([[math.cos(theta), 0, math.sin(theta)],
                        [0, 1, 0],
                        [-math.sin(theta), 0, math.cos(theta)]])

    def rotate_x(self, theta):
        return np.array([[1, 0, 0],
                        [0, math.cos(theta), -math.sin(theta)],
                        [0, math.sin(theta), math.cos(theta)]])

    def rotate_z(self, theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0],
                        [math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]])
    

    def pub_marker(self, coord3d_mat, persons): 
        LIMBS = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                        [4, 5], [5, 6], [6, 7],
                        [3, 8], [8, 9], [9, 10], [10, 11],
                        [0, 16], [16, 17], [17, 18], [18, 19], 
                        [0, 12], [12, 13], [13, 14], [14, 15],
                        [3, 20], [20, 21], [21, 22], [22, 23],
                        [21, 24], [24, 25]])

        KEYPOINT_NAME_DICT = {0: "BWaist", 1: "MWaist",
                            2: "TWaist", 3: "MNeck", 4: "RNeck",
                            5: "RShoulder", 6: "RElbow", 7: "RWrist",
                            8: "LNeck", 9: "LShoulder", 10: "LElbow",
                            11: "LWaist", 12: "LHip", 13: "ULLeg",
                            14: "BLLeg", 15: "LAnkle", 16: "RHip", 17: "URLeg",
                            18:"BRLeg", 19:"RAnkle", 20:"Chin", 21:'Nose',
                            22:"LEye", 23:"LEar", 24:"REye", 25:"REar"}
        
        for pid in range(coord3d_mat.shape[0]):
            ma = MarkerArray()
            h = Header(frame_id=self.camera_frame_id)
            line_list = Marker(type=Marker.LINE_LIST, id=pid)
            line_list.header = h
            line_list.action = Marker.ADD
            line_list.scale.x = 0.01

            vis_mat = np.ones((2, 26))
            colors = {0:[1, 0, 0], 1:[0, 1, 0]}
            for kid in range(coord3d_mat.shape[1]):
                if (np.isnan(coord3d_mat[pid, kid, 0])) or (np.isnan(coord3d_mat[pid, kid, 1])) \
                    or (np.isnan(coord3d_mat[pid, kid, 2])):
                    vis_mat[pid, kid] = 0
                else:
                    self.br.sendTransform((coord3d_mat[pid, kid, 0], coord3d_mat[pid, kid, 1], coord3d_mat[pid, kid, 2]),
                        transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "/human_pose/person%d/%s" % (pid, KEYPOINT_NAME_DICT[kid]),
                        self.camera_frame_id)

            # draw skeleton figure
            for lid, (p0, p1) in enumerate(LIMBS):
                if vis_mat[pid, p0] and vis_mat[pid, p1]:
                    p0 = Point(x=coord3d_mat[pid, p0, 0],
                            y=coord3d_mat[pid, p0, 1],
                            z=coord3d_mat[pid, p0, 2])
                    p1 = Point(x=coord3d_mat[pid, p1, 0],
                            y=coord3d_mat[pid, p1, 1],
                            z=coord3d_mat[pid, p1, 2])
                    line_list.points.append(p0)
                    line_list.points.append(p1)

            #gaze
            # if vis_mat[pid, 24] and vis_mat[pid, 22] and vis_mat[pid, 21]:
            #     a = coord3d_mat[pid, 21, :] - coord3d_mat[pid, 24, :]
            #     b = coord3d_mat[pid, 22, :] - coord3d_mat[pid, 24, :]
            #     normal = np.cross(a, b)
            #     if np.linalg.norm(normal) >0:
            #         normal = normal/np.linalg.norm(normal)
            #     normal = normal + np.array([0, 0.8, 0])
            #     center = np.vstack([coord3d_mat[pid, 24, :], coord3d_mat[pid, 21, :], coord3d_mat[pid, 22, :]]).mean(axis=0)
            #     p0 = Point(x = center[0],
            #                 y = center[1],
            #                 z = center[2])
            #     p1 = Point(x = center[0] + normal[0]*5,
            #                 y = center[1] + normal[1]*5,
            #                 z = center[2] + normal[2]*5)
            #     line_list.points.append(p0)
            #     line_list.points.append(p1)
            
            line_list.color.r = colors[pid][0]
            line_list.color.g = colors[pid][1]
            line_list.color.b = colors[pid][2]
            line_list.color.a = 1.0
            ma.markers.append(line_list)
            self.personspub[pid].publish(ma)
            
            #hand
            hands = np.zeros((2, 4, 3))
            if vis_mat[pid, 6] and vis_mat[pid, 7]:
                hands[pid, 0, 0] = 1
                x_l, y_l, z_l = coord3d_mat[pid, 6]
                x_r, y_r, z_r = coord3d_mat[pid, 7]
                normal = np.array([x_r-x_l, y_r-y_l, z_r-z_l])
                if np.linalg.norm(normal)>0:
                    normal = normal/np.linalg.norm(normal)
                hand_x = coord3d_mat[pid, 7, 0] + normal[0]*self.magnitude
                hand_y = coord3d_mat[pid, 7, 1] + normal[1]*self.magnitude
                hand_z = coord3d_mat[pid, 7, 2] + normal[2]*self.magnitude
                hands[pid, 1, :] = np.array([hand_x, hand_y, hand_z])

            if vis_mat[pid, 10] and vis_mat[pid, 11]:
                hands[pid, 0, 1] = 1
                x_l, y_l, z_l = coord3d_mat[pid, 10]
                x_r, y_r, z_r = coord3d_mat[pid, 11]
                normal = np.array([x_r-x_l, y_r-y_l, z_r-z_l])
                if np.linalg.norm(normal)>0:
                    normal = normal/np.linalg.norm(normal)
                hand_x = coord3d_mat[pid, 11, 0] + normal[0]*self.magnitude
                hand_y = coord3d_mat[pid, 11, 1] + normal[1]*self.magnitude
                hand_z = coord3d_mat[pid, 11, 2] + normal[2]*self.magnitude
                hands[pid, 2, :] = np.array([hand_x, hand_y, hand_z])

            if vis_mat[pid, 21]:
                hands[pid, 0, 2] = 1
                hands[pid, 3, :] = coord3d_mat[pid, 21, :]

            eyes = np.zeros((2, 3, 3))
            if vis_mat[pid, 22]:
                eyes[pid, 0, 0] = 1
                eyes[pid, 1, :] = coord3d_mat[pid, 22, :]
            if vis_mat[pid, 24]:
                eyes[pid, 0, 1] = 1
                eyes[pid, 2, :] = coord3d_mat[pid, 24, :]

            if persons[pid]:
                frame_id = 'handpose'+str(pid)
                msg = self.xyz_array_to_pointcloud2(hands[pid], stamp=rospy.get_rostime(), frame_id=frame_id)
                self.handspub[pid].publish(msg)
                self.br.sendTransform((0, 0, 0),
                    transformations.quaternion_from_euler(0, 0, 0),
                    rospy.Time.now(),
                    frame_id,
                    self.camera_frame_id)

                frame_id = 'eyepose'+str(pid)
                msg = self.xyz_array_to_pointcloud2(eyes[pid], stamp=rospy.get_rostime(), frame_id=frame_id)
                self.eyespub[pid].publish(msg)
                self.br.sendTransform((0, 0, 0),
                    transformations.quaternion_from_euler(0, 0, 0),
                    rospy.Time.now(),
                    frame_id,
                    self.camera_frame_id)


    def listener(self):

        while True:
            data = self.s.recv(self.coordi_num*4)
            persons = np.zeros(2)
            if(len(data)>0):
                persons[0] = unpack('f', data[:4])[0]
                persons[1] = unpack('f', data[4:8])[0]
                point_clouds = np.zeros((2, 26, 3))
                count = 0 
                for i in range(12,3*26*4 + 12,12):      
                    x = unpack('f', data[i:i+4])
                    y = unpack('f', data[i+4:i+8])
                    z = unpack('f', data[i+8:i+12])
                    point_clouds[0, count, :] = np.array([x[0]/1000.0,y[0]/1000.0,z[0]/1000.0])
                    count += 1
                    
                count = 0
                for i in range(3*26*4 + 12,self.coordi_num*4,12):        
                    x = unpack('f', data[i:i+4])
                    y = unpack('f', data[i+4:i+8])
                    z = unpack('f', data[i+8:i+12])
                    point_clouds[1, count, :] = np.array([x[0]/1000.0,y[0]/1000.0,z[0]/1000.0])
                    count += 1

                self.pub_marker(point_clouds, persons)
                
                



""" Main """
if __name__ == '__main__':
    coor_pub = CoordinatePublisher()

    try:
        while not rospy.is_shutdown():
            coor_pub.listener()

    except KeyboardInterrupt:
        coor_pub.s.close()
        print("Shutting down")
