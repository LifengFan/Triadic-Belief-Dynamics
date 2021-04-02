#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from tf import TransformBroadcaster, transformations
from cv_bridge import CvBridge, CvBridgeError
from threading import Lock
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import os
from sys import platform
import argparse
import time

# VALUES YOU MIGHT WANT TO CHANGE
COLOR_IMAGE_TOPIC = "azure/color_image"  # Ros topic of the undistorted color image
DEPTH_IMAGE_TOPIC = "azure/depth_image"  # Ros topic of the undistorted depth image
CAMERA_INFO_TOPIC = 'azure/camera_info'  # ROS topic containing camera calibration K
SKELETON1_TOPIC = 'hand_pose1'
SKELETON2_TOPIC = 'hand_pose2'
EYE1_TOPIC = 'eye_pose1'
EYE2_TOPIC = 'eye_pose2'
# NO CHANGES BEYOND THIS LINE

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["hand"] = True
params["face"] = True
params["hand_detector"] = 2
params["face_detector"] = 2
params["body"] = 0
params["num_gpu_start"] = 1

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item


class KinectDataSubscriber:
    """ Holds the most up to date """
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber(COLOR_IMAGE_TOPIC,
                                          Image,
                                          self.color_callback, queue_size=1, buff_size=10000000)
        self.depth_sub = rospy.Subscriber(DEPTH_IMAGE_TOPIC,
                                          Image,
                                          self.depth_callback, queue_size=1, buff_size=10000000)                                 
        # data containers and its mutexes
        self.color_image = None
        self.depth_image = None
        self.color_mutex = Lock()
        self.depth_mutex = Lock()

    def color_callback(self, data):
        """ Called whenever color data is available. """
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.color_mutex.acquire()
        self.color_image = cv_image
        self.color_mutex.release()

    def depth_callback(self, data):
        """ Called whenever color data is available. """
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        self.depth_mutex.acquire()
        self.depth_image = cv_image
        self.depth_mutex.release()

class HandSubscriber:
    def __init__(self):
        self.hand_sub1 = rospy.Subscriber(SKELETON1_TOPIC,
                                          PointCloud2,
                                          self.hand_callback1, queue_size=1)
        self.hand_sub2 = rospy.Subscriber(SKELETON2_TOPIC,
                                          PointCloud2,
                                          self.hand_callback2, queue_size=1)
        self.hand_box1 = None
        self.hand_box2 = None
        self.face_box1 = None
        self.face_box2 = None
        self.hand1_mutex = Lock()
        self.hand2_mutex = Lock()

    def hand_callback1(self, data):
        """ Called whenever color data is available. """
        print('hand1 received')
        self.hand1_mutex.acquire()
        hand_points = pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
        hand_points = np.array(list(hand_points))
        self.hand_box1 = [[0, 0, 0], [0, 0, 0]]
        if hand_points[0, 0]:
            self.hand_box1[0] = list(hand_points[1, :])
        if hand_points[0, 1]:
            self.hand_box1[1] = list(hand_points[2, :])
        if hand_points[0, 2]:
            self.face_box1 = list(hand_points[3, :])
        self.hand1_mutex.release()

    def hand_callback2(self, data):
        """ Called whenever color data is available. """
        print('hand2 received')
        self.hand2_mutex.acquire()
        hand_points = pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
        hand_points = np.array(list(hand_points))
        self.hand_box2 = [[0, 0, 0], [0, 0, 0]]
        if hand_points[0, 0]:
            self.hand_box2[0] = list(hand_points[1, :])
        if hand_points[0, 1]:
            self.hand_box2[1] = list(hand_points[2, :])
        if hand_points[0, 2]:
            self.face_box2 = list(hand_points[3, :])
        self.hand2_mutex.release()

class EyeSubscriber:
    def __init__(self):
        self.eye_sub1 = rospy.Subscriber(EYE1_TOPIC,
                                          PointCloud2,
                                          self.eye_callback1, queue_size=1)
        self.eye_sub2 = rospy.Subscriber(EYE2_TOPIC,
                                          PointCloud2,
                                          self.eye_callback2, queue_size=1)
        self.eye_box1 = None
        self.eye_box2 = None
        self.eye1_mutex = Lock()
        self.eye2_mutex = Lock()

    def eye_callback1(self, data):
        """ Called whenever color data is available. """
        print('eye1 received')
        self.eye1_mutex.acquire()
        eye_points = pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
        eye_points = list(eye_points)
        eye_points = np.array(eye_points)
        # print(eye_points.shape)
        self.eye_box1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        if eye_points[0, 0]:
            self.eye_box1[0][0] = 1 
            self.eye_box1[1] = list(eye_points[1, :])
        if eye_points[0][1]:
            self.eye_box1[0][1] = 1
            self.eye_box1[2] = list(eye_points[2, :])
        self.eye1_mutex.release()

    def eye_callback2(self, data):
        """ Called whenever color data is available. """
        print('eye2 received')
        self.eye2_mutex.acquire()
        eye_points = pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
        eye_points = list(eye_points)
        eye_points = np.array(eye_points)
        self.eye_box2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        if eye_points[0, 0]:
            self.eye_box2[0][0] = 1
            self.eye_box2[1] = list(eye_points[1, :])
        if eye_points[0][1]:
            self.eye_box2[0][1] = 1
            self.eye_box2[2] = list(eye_points[2, :])
        self.eye2_mutex.release()

class HandRecog:
    def __init__(self, box_size = 150, face_size = 50):
        self.box_size = box_size
        self.face_size = face_size

    def point2screen(self, points, K):
        rot_points = np.array(points)
        points_camera = rot_points.reshape(3,1)
        project_matrix = np.array(K).reshape(3,3)
        points_prj = project_matrix.dot(points_camera)
        points_prj = points_prj.transpose()
        points_prj[:,0] = points_prj[:,0]/points_prj[:,2]
        points_prj[:,1] = points_prj[:,1]/points_prj[:,2]
        points_screen = points_prj[:,:2]
        assert points_screen.shape==(1,2)
        points_screen = points_screen.reshape(-1)
        return points_screen


    def hand_recog(self, boxes, img, K, faces):
        print("box received")
        img_draw = img.copy()
        rectangles = []
        for hands in boxes:
            rectangle = []
            if hands:
                for hand in hands:
                    points_screen = self.point2screen(hand, K)

                    if (int(points_screen[1]) + self.box_size > img.shape[0]) or (int(points_screen[0]) + self.box_size > img.shape[1]) \
                        or (int(points_screen[1]) - self.box_size <0) or (int(points_screen[0]) - self.box_size < 0):
                        rectangle.append(op.Rectangle(0.0, 0.0, 0.0, 0.0))
                    else:
                        cv2.rectangle(img_draw, (int(points_screen[0])-self.box_size, int(points_screen[1])-self.box_size), 
                                    (int(points_screen[0])+self.box_size, int(points_screen[1])+self.box_size), (255,0,0), 2)
                        rectangle.append(op.Rectangle(points_screen[0]-self.box_size, points_screen[1]-self.box_size, self.box_size*2, self.box_size*2))
            else:
                rectangle.append(op.Rectangle(0.0, 0.0, 0.0, 0.0))
                rectangle.append(op.Rectangle(0.0, 0.0, 0.0, 0.0))

            assert len(rectangle) == 2
            rectangles.append(rectangle)
        assert len(rectangles) == 2
        
        # faces
        face_rectangles = []
        for face in faces:
            if face:
                points_screen = self.point2screen(face, K)

                if (int(points_screen[1]) + self.face_size > img.shape[0]) or (int(points_screen[0]) + self.face_size > img.shape[1]) \
                    or (int(points_screen[1]) - self.face_size <0) or (int(points_screen[0]) - self.face_size < 0):
                    face_rectangles.append(op.Rectangle(0.0, 0.0, 0.0, 0.0))
                else:
                    cv2.rectangle(img_draw, (int(points_screen[0])-self.face_size, int(points_screen[1])-self.face_size), 
                                    (int(points_screen[0])+self.face_size, int(points_screen[1])+self.face_size), (0,255,0), 2)
                    face_rectangles.append(op.Rectangle(points_screen[0]-self.face_size, points_screen[1]-self.face_size, self.face_size*2, self.face_size*2))
            else:
                face_rectangles.append(op.Rectangle(0.0, 0.0, 0.0, 0.0))
        assert len(face_rectangles) == 2


        cv2.imshow("hand", img_draw)
        cv2.waitKey(30)
        return rectangles, face_rectangles
            
class HandPublisher:
    def __init__(self):
        hand11_pub = rospy.Publisher('hand_openpose11', MarkerArray, queue_size=1)
        hand12_pub = rospy.Publisher('hand_openpose12', MarkerArray, queue_size=1)
        hand21_pub = rospy.Publisher('hand_openpose21', MarkerArray, queue_size=1)
        hand22_pub = rospy.Publisher('hand_openpose22', MarkerArray, queue_size=1)
        eyepub1 = rospy.Publisher('eye_pose_1', PointCloud2, queue_size=1)
        eyepub2 = rospy.Publisher('eye_pose_2', PointCloud2, queue_size=1)
        gazepub1 = rospy.Publisher('gaze_pose1', MarkerArray, queue_size=1)
        gazepub2 = rospy.Publisher('gaze_pose2', MarkerArray, queue_size=1)
        self.gaze_pub = [gazepub1, gazepub2]
        self.eyepubs = [eyepub1, eyepub2]
        self.pubs = [[hand11_pub, hand12_pub], [hand21_pub, hand22_pub]]
        self.br = TransformBroadcaster()
        self.camera_frame_id = 'azure/color_image'


    def hand2marker(self, keypoints, color, depth, K):
        """ Publishes detected persons as ROS line lists. """
        # LIMBS = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
        #                 [0, 5], [5, 6], [6, 7], [7, 8],
        #                 [0, 9], [9, 10], [10, 11], [11, 12], 
        #                 [0, 13], [13, 14], [14, 15], [15, 16],
        #                 [0, 17], [17, 18], [18, 19], [19, 20]])

        LIMBS = np.array([[0, 1], [1, 2], [2, 3], 
                        [0, 5], [5, 6], [6, 7], 
                        [0, 9], [9, 10], [10, 11],
                        [0, 13], [13, 14], [14, 15], 
                        [0, 17], [17, 18], [18, 19]])

        KEYPOINT_NAME_DICT = {0: "palm", 
                            1: "finger1_1", 2: "finger1_2", 3: "finger1_3", 4: "finger1_4",
                            5: "finger2_1", 6: "finger2_2", 7: "finger2_3", 8: "finger2_4", 
                            9: "finger3_1", 10: "finger3_2", 11: "finger3_3", 12: "finger3_4",
                            13: "finger4_1", 14: "finger4_2", 15: "finger4_3", 16: "finger4_4",
                            17: "finger5_1", 18: "finger5_2", 19: "finger5_3", 20: "finger5_4"}
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        fx_d, fy_d, cx_d, cy_d = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        for hand_id, hand in enumerate(keypoints): 
            for pid in range(hand.shape[0]):
                # broadcast keypoints as tf
                ma = MarkerArray()
                h = Header(frame_id=self.camera_frame_id)
                line_list = Marker(type=Marker.LINE_LIST, id=0)
                line_list.header = h
                line_list.action = Marker.ADD
                line_list.scale.x = 0.01
                coordinates = []
                confidence = []
                for kid in range(hand.shape[1]):
                    x_d = int(hand[pid, kid, 0])
                    y_d = int(hand[pid, kid, 1])
                    
                    if y_d < depth.shape[0] and x_d < depth.shape[1]:
                        depth_value = depth[y_d, x_d]
                        x = (x_d - cx_d) * depth_value / fx_d
                        y = (y_d - cy_d) * depth_value / fy_d
                        z = depth_value
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            x,y,z = 0,0,0
                            coordinates.append([0, 0, 0])
                            confidence.append(0.0)
                        else:
                            coordinates.append([x, y, z])
                            confidence.append(hand[pid, kid, 2])
                    else:
                        x,y,z = 0,0,0
                        coordinates.append([0, 0, 0])
                        confidence.append(0.0)
                    
                    self.br.sendTransform((x, y, z),
                        transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "/hand_pose/person%d/%s" % (pid, KEYPOINT_NAME_DICT[kid]),
                        self.camera_frame_id)

                # draw skeleton figure
                for lid, (p0, p1) in enumerate(LIMBS):
                    if confidence[p0] > 0.1 and confidence[p1] > 0.1:
                        p0 = Point(x=coordinates[p0][0],
                                y=coordinates[p0][1],
                                z=coordinates[p0][2])
                        p1 = Point(x=coordinates[p1][0],
                                y=coordinates[p1][1],
                                z=coordinates[p1][2])
                        line_list.points.append(p0)
                        line_list.points.append(p1)
                
                line_list.color.r = colors[pid][0]
                line_list.color.g = colors[pid][1]
                line_list.color.b = colors[pid][2]
                line_list.color.a = 1.0
                ma.markers.append(line_list)
                self.pubs[pid][hand_id].publish(ma)

    def check_nan(self, point):
        return np.isnan(point[0]) or np.isnan(point[1])

    def img2coor(self, img, K, color, depth):
        fx_d, fy_d, cx_d, cy_d = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_d = int(img[0])
        y_d = int(img[1])
        if y_d < depth.shape[0] and x_d < depth.shape[1]:
            depth_value = depth[y_d, x_d]
            x = (x_d - cx_d) * depth_value / fx_d
            y = (y_d - cy_d) * depth_value / fy_d
            z = depth_value
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                x,y,z = 0,0,0
        else:
            x,y,z = 0,0,0
        return [x, y - 0.05, z - 0.1]

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


    def eye2marker(self, keypoints, color, depth, K, eyes):
        """ Publishes detected persons as ROS line lists. """
        indices = [[36, 37, 38, 39, 40, 41, 68], [42, 43, 44, 45, 46, 47, 69]]
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        # indices = [[68], [69]]
        for pid in range(keypoints.shape[0]):
            left_count = 0
            right_count = 0
            left_eye = np.zeros((1, 2))
            right_eye = np.zeros((1, 2))
            for point_id, index in enumerate(indices[0]):
                if not self.check_nan(keypoints[pid, index, :2]):
                    left_count += 1
                    left_eye = np.vstack([left_eye, keypoints[pid, index, :2]])
                if not self.check_nan(keypoints[pid, indices[1][point_id], :2]):
                    right_count += 1
                    right_eye = np.vstack([right_eye, keypoints[pid, indices[1][point_id], :2]])
            
            eye_points = []
            if left_count == 7:
                left_center = left_eye[1:].mean(axis = 0)
                left_coor = self.img2coor(left_center, K, color, depth)
                eye_points.append(left_coor)
            if right_count == 7:
                right_center = right_eye[1:].mean(axis = 0)
                right_coor = self.img2coor(right_center, K, color, depth)
                eye_points.append(right_coor)
            
            # msg = self.xyz_array_to_pointcloud2(np.array(eye_points), stamp=rospy.get_rostime(), frame_id='eyepose1')
            # self.eyepubs[pid].publish(msg)
            # self.br.sendTransform((0, 0, 0),
            #     transformations.quaternion_from_euler(0, 0, 0),
            #     rospy.Time.now(),
            #     'eyepose1',
            #     self.camera_frame_id)

            ma = MarkerArray()
            h = Header(frame_id=self.camera_frame_id)
            line_list = Marker(type=Marker.LINE_LIST, id=0)
            line_list.header = h
            line_list.action = Marker.ADD
            line_list.scale.x = 0.01

            if eyes[pid]:
                if eyes[pid][0] and eyes[pid][1]:
                    eyes_center = (np.array(eyes[pid][1])  + np.array(eyes[pid][2]))/2
                    normal = np.array([0, 0, 0])
                    if left_count == 7 and right_count == 7:
                        avg = np.array(eye_points).mean(axis = 0)
                        normal = avg - eyes_center
                    elif left_count == 7:
                        assert len(eye_points) == 1
                        normal = np.array(eye_points[0]) - eyes[1, :]
                    elif right_count == 7:
                        assert len(eye_points) == 1
                        normal = np.array(eye_points[1]) - eyes[2, :]
                    if np.linalg.norm(normal) > 0:
                        normal = normal/np.linalg.norm(normal)
                    print(normal)
                    p0 = Point(x=eyes_center[0],
                            y=eyes_center[1],
                            z=eyes_center[2])
                    p1 = Point(x=eyes_center[0] + normal[0]*2,
                            y=eyes_center[1] + normal[1]*2,
                            z=eyes_center[2] + normal[2]*2)
                    line_list.points.append(p0)
                    line_list.points.append(p1)
                
                    line_list.color.r = colors[pid][0]
                    line_list.color.g = colors[pid][1]
                    line_list.color.b = colors[pid][2]
                    line_list.color.a = 1.0
                    ma.markers.append(line_list)
                    self.gaze_pub[pid].publish(ma)
 

class CameraCalibSubscriber():
    def __init__(self, camera_info_topic):
        self.subscriber = rospy.Subscriber(camera_info_topic,
                                        CameraInfo, self.camera_callback, queue_size=1)
        self.stop = False
        self.K = None
        self.camera_frame_id = None

    def camera_callback(self, data):
        self.K = np.reshape(np.array(data.K), [3, 3])
        self.camera_frame_id = data.header.frame_id
        self.stop = True

    def wait_for_calib(self):
        try:
            while not self.stop:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down")

        return self.K, self.camera_frame_id


""" Main """
if __name__ == '__main__':
    # read calib from ros topic
    camera_calib = CameraCalibSubscriber(CAMERA_INFO_TOPIC)
    rospy.init_node('listener', anonymous=True)
    K, camera_frame_id = camera_calib.wait_for_calib()

    # Start Node that reads the kinect topic
    image_data = KinectDataSubscriber()
    hand_data = HandSubscriber()
    eye_data = EyeSubscriber()
    rospy.init_node('listener', anonymous=True) 

    hand_pub = HandPublisher()
    recognizer = HandRecog()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # loop
    try:
        while not rospy.is_shutdown():
            image_data.color_mutex.acquire()
            image_data.depth_mutex.acquire()
            hand_data.hand1_mutex.acquire()
            hand_data.hand2_mutex.acquire()
        
            if (image_data.color_image is not None) and (image_data.depth_image is not None):
                color = image_data.color_image.copy()
                depth = image_data.depth_image.copy()
                boxes = [hand_data.hand_box1, hand_data.hand_box2]
                faces = [hand_data.face_box1, hand_data.face_box2]
                eyes = [eye_data.eye_box1, eye_data.eye_box2]

                image_data.color_mutex.release()
                image_data.depth_mutex.release()
                hand_data.hand1_mutex.release()
                hand_data.hand2_mutex.release()
                rectangles, face_rectangles = recognizer.hand_recog(boxes, color, K, faces)
                datum = op.Datum()
                datum.cvInputData = color
                datum.handRectangles = rectangles
                datum.faceRectangles = face_rectangles
                # Process and display image
                opWrapper.emplaceAndPop([datum])
                hand_pub.hand2marker(datum.handKeypoints, color, depth, K)
                hand_pub.eye2marker(datum.faceKeypoints, color, depth, K, eyes)
                cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
                cv2.waitKey(100)
                

            else:
                image_data.color_mutex.release()
                image_data.depth_mutex.release()
                hand_data.hand1_mutex.release()
                hand_data.hand2_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
