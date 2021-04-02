#!/usr/bin/env python
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import rospy
from tf import TransformBroadcaster, transformations
import tf
import joblib
import glob
import os
import pickle
import numpy as np
import argparse
import math
from pyquaternion import Quaternion


import cv2


class SkeleSmooth:
    def __init__(self):
        self.pub1 = rospy.Publisher('skele1_test', PointCloud2, queue_size=1)
        self.pub2 = rospy.Publisher('skele2_test', PointCloud2, queue_size=1)
        self.obj_pub = rospy.Publisher('objs', PointCloud2, queue_size=1)
        self.br = TransformBroadcaster()
        rospy.init_node('talker', anonymous=True)
        self.listener = tf.TransformListener()
        self.trackers = {'test1': 'skele1.p', 'test2': 'skele2.p', 'test6': 'skele2.p', 'test7': 'skele1.p',
                         'test_9434_1': 'skele2.p', 'test_9434_3': 'skele2.p', 'test_9434_18': 'skele1.p',
                         'test_94342_0': 'skele2.p', 'test_94342_1': 'skele2.p', 'test_94342_2': 'skele2.p',
                         'test_94342_3': 'skele2.p', 'test_94342_4': 'skele1.p', 'test_94342_5': 'skele1.p',
                         'test_94342_6': 'skele1.p', 'test_94342_7': 'skele1.p', 'test_94342_8': 'skele1.p',
                         'test_94342_10': 'skele2.p', 'test_94342_11': 'skele2.p', 'test_94342_12': 'skele1.p',
                         'test_94342_13': 'skele2.p', 'test_94342_14': 'skele1.p', 'test_94342_15': 'skele2.p',
                         'test_94342_16': 'skele1.p', 'test_94342_17': 'skele2.p', 'test_94342_18': 'skele1.p',
                         'test_94342_19': 'skele2.p', 'test_94342_20': 'skele1.p', 'test_94342_21': 'skele2.p',
                         'test_94342_22': 'skele1.p', 'test_94342_23': 'skele1.p', 'test_94342_24': 'skele1.p',
                         'test_94342_25': 'skele2.p', 'test_94342_26': 'skele1.p',
                         'test_boelter_1': 'skele2.p', 'test_boelter_2': 'skele2.p', 'test_boelter_3': 'skele2.p',
                         'test_boelter_4': 'skele1.p', 'test_boelter_5': 'skele1.p', 'test_boelter_6': 'skele1.p',
                         'test_boelter_7': 'skele1.p', 'test_boelter_9': 'skele1.p', 'test_boelter_10': 'skele1.p',
                         'test_boelter_12': 'skele2.p', 'test_boelter_13': 'skele1.p', 'test_boelter_14': 'skele1.p',
                         'test_boelter_15': 'skele1.p', 'test_boelter_17': 'skele2.p', 'test_boelter_18': 'skele1.p',
                         'test_boelter_19': 'skele2.p', 'test_boelter_21': 'skele1.p', 'test_boelter_22': 'skele2.p',
                         'test_boelter_24': 'skele1.p', 'test_boelter_25': 'skele1.p',
                         'test_boelter2_0': 'skele1.p', 'test_boelter2_2': 'skele1.p', 'test_boelter2_3': 'skele1.p',
                         'test_boelter2_4': 'skele1.p', 'test_boelter2_5': 'skele1.p', 'test_boelter2_6': 'skele1.p',
                         'test_boelter2_7': 'skele2.p', 'test_boelter2_8': 'skele2.p', 'test_boelter2_12': 'skele2.p',
                         'test_boelter2_14': 'skele2.p', 'test_boelter2_15': 'skele2.p', 'test_boelter2_16': 'skele1.p',
                         'test_boelter2_17': 'skele1.p',
                         'test_boelter3_0': 'skele1.p', 'test_boelter3_1': 'skele2.p', 'test_boelter3_2': 'skele2.p',
                         'test_boelter3_3': 'skele2.p', 'test_boelter3_4': 'skele1.p', 'test_boelter3_5': 'skele2.p',
                         'test_boelter3_6': 'skele2.p', 'test_boelter3_7': 'skele1.p', 'test_boelter3_8': 'skele2.p',
                         'test_boelter3_9': 'skele2.p', 'test_boelter3_10': 'skele1.p', 'test_boelter3_11': 'skele2.p',
                         'test_boelter3_12': 'skele2.p', 'test_boelter3_13': 'skele2.p',
                         'test_boelter4_0': 'skele2.p', 'test_boelter4_1': 'skele2.p', 'test_boelter4_2': 'skele2.p',
                         'test_boelter4_3': 'skele2.p', 'test_boelter4_4': 'skele2.p', 'test_boelter4_5': 'skele2.p',
                         'test_boelter4_6': 'skele2.p', 'test_boelter4_7': 'skele2.p', 'test_boelter4_8': 'skele2.p',
                         'test_boelter4_9': 'skele2.p', 'test_boelter4_10': 'skele2.p', 'test_boelter4_11': 'skele2.p',
                         'test_boelter4_12': 'skele2.p', 'test_boelter4_13': 'skele2.p',
                         }

    def reseq(self, img_path='./post_images/kinect/', save_prefix='./post_skeletons/'):
        img_folders = os.listdir(img_path)

        for img_folder in img_folders:
            # print('**********' + img_path + img_folder + '************')
            clips = sorted(glob.glob(img_path + img_folder + '/*'))
            for clip in clips:
                clip_name = clip.split('/')[-1]
                if not clip_name == "test_boelter2_5":
                    continue
                print('************' + clip + '****************')
                save_path = save_prefix + clip_name
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # else:
                #     continue

                with open(clip + '/skele1.p', 'rb') as f:
                    skeles1_input = pickle.load(f)
                with open(clip + '/skele2.p', 'rb') as f:
                    skeles2_input = pickle.load(f)

                skele1 = None
                skele2 = None
                for i in range(len(skeles1_input)):
                    if i == 0:
                        skele1 = []
                        skele2 = []
                        skele1.append([[0, 0, 0]] * 26)
                        skele2.append([[0, 0, 0]] * 26)

                    # if i>0 and i<60:
                    #     skele1.append([[0, 0, 0]]*26)
                    #     skele2.append([[0, 0, 0]]*26)

                    # if skele1 == None:
                    #     skele1 = []
                    #     skele2 = []
                    #     skele1.append(skeles1_input[i])
                    #     skele2.append(skeles2_input[i])
                    else:
                        if np.linalg.norm(np.array(skeles2_input[i]) - np.array(skeles1_input[i])) < 1:
                            temp = []
                            temp.append(np.linalg.norm(np.array(skele1[i - 1]) - np.array(skeles1_input[i])))
                            temp.append(np.linalg.norm(np.array(skele2[i - 1]) - np.array(skeles1_input[i])))
                            idx = np.argmin(np.array(temp))
                            if idx == 0:
                                skele1.append(skeles1_input[i])
                                skele2.append(skele2[i - 1])
                            else:
                                skele2.append(skeles1_input[i])
                                skele1.append(skele1[i - 1])
                        elif np.mean(np.array(skeles1_input[i])) == 0 and np.mean(np.array(skeles2_input[i])) == 0:
                            skele2.append(skele2[i - 1])
                            skele1.append(skele1[i - 1])
                        elif np.mean(np.array(skeles1_input[i])) == 0:
                            temp = []
                            temp.append(np.linalg.norm(np.array(skele1[i - 1]) - np.array(skeles2_input[i])))
                            temp.append(np.linalg.norm(np.array(skele2[i - 1]) - np.array(skeles2_input[i])))
                            idx = np.argmin(np.array(temp))
                            if idx == 0:
                                skele1.append(skeles2_input[i])
                                skele2.append(skele2[i - 1])
                            else:
                                skele2.append(skeles2_input[i])
                                skele1.append(skele1[i - 1])
                        elif np.mean(np.array(skeles2_input[i])) == 0:
                            temp = []
                            temp.append(np.linalg.norm(np.array(skele1[i - 1]) - np.array(skeles1_input[i])))
                            temp.append(np.linalg.norm(np.array(skele2[i - 1]) - np.array(skeles1_input[i])))
                            idx = np.argmin(np.array(temp))

                            # if i > 160:
                            #     print(i)
                            # # print(skeles2_input[i])
                            # # print(skeles1_input[i])
                            # # print(skele1[i-1])
                            # # print(np.linalg.norm(np.array(skeles2_input[i]) - np.array(skeles1_input[i])))
                            #     raw_input("Press Enter to continue...")
                            #     print(skeles1_input[i])
                            #     print(skeles2_input[i])
                            #     print(skele1[i-1])
                            #     print(skele2[i-1])
                            #     print(temp)

                            if idx == 0:
                                skele1.append(skeles1_input[i])
                                skele2.append(skele2[i - 1])
                            else:
                                skele2.append(skeles1_input[i])
                                skele1.append(skele1[i - 1])
                        else:
                            temp = []
                            temp.append(np.linalg.norm(np.array(skele1[i - 1]) - np.array(skeles1_input[i])))
                            temp.append(np.linalg.norm(np.array(skele1[i - 1]) - np.array(skeles2_input[i])))
                            idx = np.argmin(np.array(temp))

                            # if idx == 0 and temp[idx] < 7:
                            #     skele1.append(skeles1_input[i])
                            # elif idx == 1 and temp[idx] < 7:
                            #     skele1.append(skeles2_input[i])
                            # else:
                            #     skele1.append(skele1[i-1])
                            # if i>920:
                            #     print(i)
                            #     raw_input("Press Enter to continue...")
                            #     print(skeles1_input[i])
                            #     print(skeles2_input[i])
                            #     print(skele1[i-1])
                            #     print(skele2[i-1])
                            #     print(temp)
                            if idx == 0 and temp[idx] < 50:
                                skele1.append(skeles1_input[i])
                                skele2.append(skeles2_input[i])
                            elif idx == 1 and temp[idx] < 50:
                                skele1.append(skeles2_input[i])
                                skele2.append(skeles1_input[i])
                            else:
                                skele1.append(skele1[i - 1])
                                skele2.append(skele2[i - 1])

                            temp = []
                            temp.append(np.linalg.norm(np.array(skele2[i - 1]) - np.array(skeles1_input[i])))
                            temp.append(np.linalg.norm(np.array(skele2[i - 1]) - np.array(skeles2_input[i])))
                            idx = np.argmin(np.array(temp))

                            # if idx == 0 and temp[idx] < 50:
                            #     skele2.append(skeles1_input[i])
                            #     skele1.append(skeles2_input[i])
                            # elif idx == 1 and temp[idx] < 50:
                            #     skele2.append(skeles2_input[i])
                            #     skele1.append(skeles1_input[i])
                            # else:
                            #     skele2.append(skele2[i-1])
                            #     skele1.append(skele1[i-1])

                            # if i > 0:
                            #     print(i)
                            # # print(skeles2_input[i])
                            # # print(skeles1_input[i])
                            # # print(skele1[i-1])
                            # # print(np.linalg.norm(np.array(skeles2_input[i]) - np.array(skeles1_input[i])))
                            # if i>150:
                            #     print(i)
                            #     raw_input("Press Enter to continue...")
                            #     print(skeles1_input[i])
                            #     print(skeles2_input[i])
                            #     print(skele1[i-1])
                            #     print(skele2[i-1])
                            #     print(temp)
                            # if idx == 0 and temp[idx] < 7:
                            #     skele2.append(skeles1_input[i])
                            # elif idx == 1 and temp[idx] < 7:
                            #     skele2.append(skeles2_input[i])
                            # else:
                            #     skele2.append(skele2[i-1])

                        # 94342_1
                        # if i == 925:
                        #     skele2[i] = skele1[i]
                        #     skele1[i] = skele1[i-1]

                        # b_15
                        # if i < 246 and i > 86:
                        #     skele2[i] = skele2[86]

                        # 3_5
                        # if i < 168 and i > 159:
                        #     skele1[i] = skele1[159]

                        # 94342_3
                        # if i == 1421:
                        #     temp = skele1[i]
                        #     skele1[i] = skele2[i]
                        #     skele2[i] = temp

                        # 94342_12
                        # if i == 906:
                        #     skele2[i] = skele1[i]
                        #     skele1[i] = skele1[i-1]

                        if i == 1538:
                            skele2[i] = skele1[i]
                            skele1[i] = skele1[i - 1]

                with open(save_path + '/skele1.p', 'wb') as f:
                    pickle.dump(skele1, f)
                with open(save_path + '/skele2.p', 'wb') as f:
                    pickle.dump(skele2, f)

    def skele_smooth(self, skele_path='./post_skeletons/', save_path='./smooth_skeletons/'):
        folders = os.listdir(skele_path)
        skeles_name = ['skele1.p', 'skele2.p']
        for folder in folders:
            if not folder == "test_boelter2_5":
                continue
            for i in range(2):
                skele_new = []
                print(skele_path + folder + '/' + skeles_name[i])
                with open(skele_path + folder + '/' + skeles_name[i], 'rb') as f:
                    skeles = pickle.load(f)
                temp_id = None
                for ske_id, skele in enumerate(skeles):
                    # if ske_id > 500:
                    #     print(ske_id)
                    #     raw_input("Press Enter to continue...")
                    #     print(temp_id)
                    skele = np.array(skele)
                    skele_new.append(skele)
                    if np.mean(skele) == 0:
                        if temp_id == None:
                            temp_id = ske_id
                        continue
                    elif not temp_id == None:
                        if ske_id - temp_id > 0 and ske_id - temp_id < 15 and temp_id > 20:
                            for j in range(temp_id, ske_id):
                                skele_new[j] = (skele_new[ske_id] - skele_new[temp_id - 1]) / (ske_id - temp_id + 1) * (
                                            j - temp_id + 1) + skele_new[temp_id - 1]
                        temp_id = None
                if not os.path.exists(save_path + folder):
                    os.makedirs(save_path + folder)
                with open(save_path + folder + '/' + skeles_name[i], 'wb') as f:
                    pickle.dump(skele_new, f)

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
        msg.row_step = 12 * points.shape[0]
        msg.is_dense = int(np.isfinite(points).all())
        msg.data = np.asarray(points, np.float32).tostring()

        return msg

    def xyzrgb_array_to_pointcloud2(self, points, colors, stamp=None, frame_id=None, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        msg = PointCloud2()
        assert (points.shape == colors.shape)

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
            # print(N)
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

    def extract_gaze(self, skeleton):
        skeleton = np.array(skeleton)
        a = skeleton[21] - skeleton[24]
        b = skeleton[22] - skeleton[24]
        normal = np.cross(a, b)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        normal = normal + np.array([0, 0.8, 0])
        gaze_center = np.vstack([skeleton[21], skeleton[22], skeleton[24]]).mean(axis=0) + np.array([0, 0.2, 0])
        return normal + gaze_center

    def extract_center(self, skeleton):
        skeleton = np.array(skeleton)
        a = skeleton[21] - skeleton[24]
        b = skeleton[22] - skeleton[24]
        normal = np.cross(a, b)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        normal = normal + np.array([0, 0.8, 0])
        gaze_center = np.vstack([skeleton[21], skeleton[22], skeleton[24]]).mean(axis=0) + np.array([0, 0.2, 0])
        return gaze_center

    def cal_angle(self, normal, obj):
        if np.linalg.norm(obj) > 0:
            obj = obj / np.linalg.norm(obj)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        cosin = obj.dot(normal)
        return cosin

    def cal_quaternion(self, gaze_center):
        theta = np.arctan2(gaze_center[2], gaze_center[0])
        return theta

    def test_skele(self, clip='test_94342_14'):
        print('../3d_pose2gaze/record_gaze_normal_new_results/' + clip + '.p')
        if not os.path.exists('../3d_pose2gaze/record_gaze_normal_new_results/' + clip + '.p'):
            return
        save_path = '../3d_pose2gaze/transformed_normal_new_results/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if os.path.exists(save_path + clip + '.p'):
            return

        with open('../../data_preprocessing2/post_skeletons/' + clip + '/skele2.p', 'rb') as f:
            skele2 = pickle.load(f)

        with open('../../data_preprocessing2/post_skeletons/' + clip + '/skele1.p', 'rb') as f:
            skele1 = pickle.load(f)

        with open('../3d_pose2gaze/record_gaze_normal_new_results/' + clip + '.p', 'rb') as f:
            gazes = joblib.load(f)

        # with open('../../data_preprocessing2/gaze360_label_reformat/' + clip + '.p', 'rb') as f:
        #     gazes_gt = joblib.load(f)
        #
        # with open('../../data_preprocessing2/gaze_training/' + clip + '/others.p', 'rb') as f:
        #     gazes_gt_ = joblib.load(f)


        # img_names = sorted(glob.glob('../data_preprocessing2/annotations/' + clip + '/kinect/*.jpg'))
        # img = cv2.imread(img_names[0])
        # cv2.imshow("img", img)
        # cv2.waitKey(500)

        rate = rospy.Rate(20)
        gaze_normal_dict = dict()
        for i in range(len(skele1)):
            print(i)
            # img = cv2.imread(img_names[i])
            # cv2.imshow("img", img)
            # cv2.waitKey(20)
            # if i > 2000:
            #     raw_input("Press Enter to continue...")
            #     print(np.linalg.norm(np.array(skele1[i]) - np.array(skele1[i-1])))
            # skele1.append(self.extract_gaze(skele1))
            with open('/home/shuwen/Downloads/pointclouds/' + clip + '/' + '{0:04}'.format(i) + '.p', 'rb') as f:
                mask = joblib.load(f)
            msg = self.xyz_array_to_pointcloud2(np.array(skele1[i]), stamp=rospy.get_rostime(), frame_id='skele1')
            self.pub1.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                  (0.7071, 0, 0, -0.7071),
                                  rospy.Time.now(),
                                  'skele1',
                                  'world')
            # rate.sleep()

            msg = self.xyz_array_to_pointcloud2(np.array(skele2[i]), stamp=rospy.get_rostime(), frame_id='skele2')
            self.pub2.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                  (0.7071, 0, 0, -0.7071),
                                  rospy.Time.now(),
                                  'skele2',
                                  'world')

            obj_points = []
            obj_colors = []
            for obj in mask:
                for sub_obj in obj[1]:
                    avg_col = np.array(sub_obj).mean(axis=1)
                    obj_center = np.array(sub_obj)[avg_col != 0, :].mean(axis=0)
                    # normal_t = obj_center - self.extract_center(skele2[i])
                    # gaze_t = self.extract_gaze(skele2[i]) - self.extract_center(skele2[i])
                    # if self.cal_angle(np.array([normal_t[0], normal_t[2]]), np.array([gaze_t[0], gaze_t[2]]))<0.5:
                    #     continue
                    # if self.cal_angle(normal_t, gaze_t) < 0.5:
                    #     continue
                    obj_points.append(obj_center)
                    obj_colors.append([1, 0, 0])



            obj_points = np.array(obj_points)
            obj_colors = np.array(obj_colors)
            if i in gazes:
                gaze_center = gazes[i][1] #+ np.array([0, -0.1, 0])
                theta = self.cal_quaternion(gaze_center)

                if theta > 0 and theta < 180:
                    euler = (0, 1.57 - theta, 0)
                else:
                    continue
                self.br.sendTransform((gaze_center[0], gaze_center[1], gaze_center[2]),
                                      tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2]),
                                      rospy.Time.now(),
                                      'eyes',
                                      'azure/color_image')

                gaze_normal = gazes[i][0]
                self.br.sendTransform((-gaze_normal[0], -gaze_normal[1], gaze_normal[2]),
                                      tf.transformations.quaternion_from_euler(0, 0, 0),
                                      rospy.Time.now(),
                                      'gaze_normal',
                                      'eyes')
            else:
                continue

            # if i in gazes_gt.keys():
            #     gaze_gt = gazes_gt[i]
            #     gaze_gt = np.array(gaze_gt)
            #     if np.linalg.norm(gaze_gt) > 0:
            #         gaze_gt = gaze_gt / np.linalg.norm(gaze_gt)
            #     self.br.sendTransform((gaze_gt[0], gaze_gt[1], gaze_gt[2]),
            #                           tf.transformations.quaternion_from_euler(0, 0, 0),
            #                           rospy.Time.now(),
            #                           'gaze_normal_gt',
            #                           'eyes')

            # if gazes_gt_[i]:
            #     gaze_gt = gazes_gt_[i][1]
            #     self.br.sendTransform((gaze_gt[0], gaze_gt[1], gaze_gt[2]),
            #                           tf.transformations.quaternion_from_euler(0, 0, 0),
            #                           rospy.Time.now(),
            #                           'gaze_normal_gt_gt',
            #                           'azure/color_image')

            msg = self.xyzrgb_array_to_pointcloud2(np.array(obj_points), np.array(obj_colors),
                                                   stamp=rospy.get_rostime(), frame_id='objs')
            self.obj_pub.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                  (0.7071, 0, 0, -0.7071),
                                  rospy.Time.now(),
                                  'objs',
                                  'world')
            try:
                (trans, rot) = self.listener.lookupTransform('azure/color_image', 'gaze_normal', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            gaze_normal_dict[i] = [[trans[0], trans[1], trans[2]], gaze_center]
            # self.br.sendTransform((trans[0], trans[1], trans[2]),
            #                       (0, 0, 0, 1),
            #                       rospy.Time.now(),
            #                       'gaze_normal_test',
            #                       'azure/color_image')
            rate.sleep()
        with open(save_path + clip + '.p', 'wb') as f:
            joblib.dump(gaze_normal_dict, f)

    def test_skele_reverse_for_inference(self, clip='test_94342_14'):
        #supposed for tracker but changed to battery
        save_path = '../../data_preprocessing2/gaze360_test_skele_input_battery/'
        if os.path.exists(save_path + clip + '.p'):
            return

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open('../../data_preprocessing2/post_skeletons/' + clip + '/skele2.p', 'rb') as f:
            skele2 = pickle.load(f)

        with open('../../data_preprocessing2/post_skeletons/' + clip + '/skele1.p', 'rb') as f:
            skele1 = pickle.load(f)

        if self.trackers[clip].split('.')[0][5] == '2':
            skele_file = '/skele1.p'
        else:
            skele_file = '/skele2.p'
        with open('../../data_preprocessing2/post_skeletons/' + clip + skele_file, 'rb') as f:
            skele2reformat = pickle.load(f)


        rate = rospy.Rate(20)
        skeleton_reformat_frames = dict()
        for i in range(len(skele1)):
            # print(i)

            msg = self.xyz_array_to_pointcloud2(np.array(skele1[i]), stamp=rospy.get_rostime(), frame_id='skele1')
            self.pub1.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                  (0.7071, 0, 0, -0.7071),
                                  rospy.Time.now(),
                                  'skele1',
                                  'world')
            # rate.sleep()

            msg = self.xyz_array_to_pointcloud2(np.array(skele2[i]), stamp=rospy.get_rostime(), frame_id='skele2')
            self.pub2.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                  (0.7071, 0, 0, -0.7071),
                                  rospy.Time.now(),
                                  'skele2',
                                  'world')

            gaze_center = self.extract_center(skele2reformat[i])
            theta = self.cal_quaternion(gaze_center)

            if theta > 0 and theta < 180:
                euler = (0, 1.57 - theta, 0)
            else:
                euler = (0, 0, 0)

            self.br.sendTransform((gaze_center[0], gaze_center[1], gaze_center[2]),
                                  tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2]),
                                  rospy.Time.now(),
                                  'eyes',
                                  'azure/color_image')

            for j in range(26):
                self.br.sendTransform((skele2reformat[i][j][0], skele2reformat[i][j][1], skele2reformat[i][j][2]),
                                      tf.transformations.quaternion_from_euler(0, 0, 0),
                                      rospy.Time.now(),
                                      "azure/human_pose_%s" % (j),
                                      'azure/color_image')

            skeleton_reformat_temp = []
            for j in range(26):
                try:
                    (trans, rot) = self.listener.lookupTransform('eyes', "azure/human_pose_%s" % (j), rospy.Time(0))
                    skeleton_reformat_temp.append([trans[0], trans[1], trans[2]])
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue


            # assert len(skeleton_reformat_temp) == 26
            if len(skeleton_reformat_temp) == 26:
                print(i)
                skeleton_reformat_frames[i] = skeleton_reformat_temp

            rate.sleep()
        # with open(save_path + clip + '.p', 'wb') as f:
        #     joblib.dump(gaze_normal_dict, f)
        with open(save_path + clip + '.p', 'wb') as f:
            joblib.dump(skeleton_reformat_frames, f)

    def test_skele_reverse(self, clip='test_94342_14'):
        # battery
        save_path = '../../data_preprocessing2/gaze360_label_reformat_with_skele_for_inference/'
        if os.path.exists(save_path + clip + '.p'):
            return

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open('../../data_preprocessing2/post_skeletons/' + clip + '/skele2.p', 'rb') as f:
            skele2 = pickle.load(f)

        with open('../../data_preprocessing2/post_skeletons/' + clip + '/skele1.p', 'rb') as f:
            skele1 = pickle.load(f)

        if self.trackers[clip].split('.')[0][5] == '2':
            skele_file = '/skele1.p'
        else:
            skele_file = '/skele2.p'

        with open('../../data_preprocessing2/post_skeletons/' + clip + '/' + skele_file, 'rb') as f:
            skele2reformat = pickle.load(f)

        with open('../../data_preprocessing2/gaze_training/' + clip + '/others.p', 'rb') as f:
            gazes = joblib.load(f)

        # img_names = sorted(glob.glob('../data_preprocessing2/annotations/' + clip + '/kinect/*.jpg'))
        # img = cv2.imread(img_names[0])
        # cv2.imshow("img", img)
        # cv2.waitKey(500)

        rate = rospy.Rate(20)
        gaze_normal_dict = dict()
        skeleton_reformat_frames = dict()
        for i in range(len(skele1)):
            print(i)
            # img = cv2.imread(img_names[i])
            # cv2.imshow("img", img)
            # cv2.waitKey(20)
            # if i > 2000:
            #     raw_input("Press Enter to continue...")
            #     print(np.linalg.norm(np.array(skele1[i]) - np.array(skele1[i-1])))
            # skele1.append(self.extract_gaze(skele1))
            with open('/home/shuwen/Downloads/pointclouds/' + clip + '/' + '{0:04}'.format(i) + '.p', 'rb') as f:
                mask = joblib.load(f)
            msg = self.xyz_array_to_pointcloud2(np.array(skele1[i]), stamp=rospy.get_rostime(), frame_id='skele1')
            self.pub1.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                  (0.7071, 0, 0, -0.7071),
                                  rospy.Time.now(),
                                  'skele1',
                                  'world')
            # rate.sleep()

            msg = self.xyz_array_to_pointcloud2(np.array(skele2[i]), stamp=rospy.get_rostime(), frame_id='skele2')
            self.pub2.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                  (0.7071, 0, 0, -0.7071),
                                  rospy.Time.now(),
                                  'skele2',
                                  'world')

            if gazes[i]:
                gaze_center = gazes[i][2] + np.array([0, -0.1, 0])
                theta = self.cal_quaternion(gaze_center)

                if theta > 0 and theta < 180:
                    euler = (0, 1.57 - theta, 0)
                else:
                    euler = (0, 0, 0)
                    # continue
                self.br.sendTransform((gaze_center[0], gaze_center[1], gaze_center[2]),
                                      tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2]),
                                      rospy.Time.now(),
                                      'eyes',
                                      'azure/color_image')

                gaze_normal = gazes[i][1]
                self.br.sendTransform((gaze_normal[0], gaze_normal[1], gaze_normal[2]),
                                      tf.transformations.quaternion_from_euler(0, 0, 0),
                                      rospy.Time.now(),
                                      'gaze_normal',
                                      'azure/color_image')
            # else:
            #     continue

            for j in range(26):
                self.br.sendTransform((skele2reformat[i][j][0], skele2reformat[i][j][1], skele2reformat[i][j][2]),
                                      tf.transformations.quaternion_from_euler(0, 0, 0),
                                      rospy.Time.now(),
                                      "azure/human_pose_%s" % (j),
                                      'azure/color_image')


            try:
                (trans, rot) = self.listener.lookupTransform('eyes', 'gaze_normal', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            # trans = np.array(trans)
            # if np.linalg.norm(trans) > 0:
            #     trans = trans/np.linalg.norm(trans)

            gaze_normal_dict[i] = [trans[0], trans[1], trans[2]]
            self.br.sendTransform((trans[0], trans[1], trans[2]),
                                  (0, 0, 0, 1),
                                  rospy.Time.now(),
                                  'gaze_normal_test',
                                  'eyes')

            skeleton_reformats = np.zeros((26, 3))
            skeleton_reformat_temp = []
            for j in range(26):
                try:
                    (trans, rot) = self.listener.lookupTransform('eyes', "azure/human_pose_%s" % (j), rospy.Time(0))
                    skeleton_reformats[j, :] = np.array([trans[0], trans[1], trans[2]])
                    skeleton_reformat_temp.append([trans[0], trans[1], trans[2]])
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue

            if len(skeleton_reformat_temp) == 26:
                skeleton_reformat_frames[i] = skeleton_reformat_temp

            rate.sleep()
        # with open(save_path + clip + '.p', 'wb') as f:
        #     joblib.dump(gaze_normal_dict, f)
        with open(save_path + clip + '.p', 'wb') as f:
            joblib.dump(skeleton_reformat_frames, f)

    def record_transformed_normal(self):
        data_path = '../../data_preprocessing2/post_skeletons/'
        clips = os.listdir(data_path)
        for clip in clips:
            print(clip)
            self.test_skele(clip)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('clip', help='test config file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    skele_smoother = SkeleSmooth()
    args = parse_args()
    # skele_smoother.reseq()
    # skele_smoother.skele_smooth()
    # skele_smoother.test_skele(args.clip)
    skele_smoother.record_transformed_normal()