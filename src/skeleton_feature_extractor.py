import pickle
import os

import joblib
import numpy as np
import glob
import cv2
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self):
        self.check_frames = [0, -5, -9, -14, -20, -27, -35, -44, -54, -65]
        self.check_frames = [0, -1] #, -2, -3, -4, -5, -6]#, -44, -54, -65]
        self.check_joints = [0, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 19, 21, 22, 24]
        self.check_head_joints = [20, 21, 22, 24]
        self.check_head_joints_no_chin = [21, 22, 24]
        self.anchor_joints = [1, 20, 7, 11]
        self.shift = [0, 0, 50, 50]
        self.box_size = [50, 50, 50, 50]
        winSize = (32, 32)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (16, 16)
        nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        # self.hog = cv2.HOGDescriptor()
        self.event_seg_battery = {
            'test_9434_18': [[0, 96, 0], [97, 361, 0], [362, 528, 0], [529, 608, 0], [609, 825, 0], [864, 1041, 0]],
            'test_94342_1': [[0, 751, 0], [752, 876, 0], [877, 1167, 0], [1168, 1386, 0]],
            'test_94342_6': [[0, 95, 0], [836, 901, 0]],
            'test_94342_10': [[0, 156, 0], [157, 169, 0], [245, 274, 0], [275, 389, 0], [390, 525, 0], [526, 665, 0],
                              [666, 680, 0]],
            'test_94342_21': [[0, 13, 0], [1098, 1133, 0]],
            'test1': [[0, 94, 0], [95, 155, 0], [156, 225, 0], [226, 559, 0], [690, 698, 0]],
            'test6': [[0, 488, 0], [489, 541, 0], [542, 672, 0], [672, 803, 0]],
            'test7': [[0, 70, 0], [71, 100, 0], [221, 226, 0]],
            'test_boelter_2': [[0, 318, 0], [319, 458, 0], [459, 543, 0], [544, 606, 0]],
            'test_boelter_7': [[0, 69, 0], [119, 133, 0], [134, 187, 0], [188, 239, 0], [329, 376, 0], [398, 491, 0],
                               [492, 564, 0], [689, 774, 0], [775, 862, 0], [863, 897, 0], [959, 1000, 0],
                               [1001, 1178, 0], [1268, 1307, 0], [1307, 1327, 0]],
            'test_boelter_24': [[0, 62, 0], [293, 314, 0]],
            'test_boelter_12': [[48, 219, 0], [220, 636, 0]],
            'test_9434_1': [[0, 67, 0], [68, 124, 0], [252, 343, 0], [344, 380, 0], [381, 417, 0]],
            'test_94342_16': [[0, 84, 0], [201, 214, 0]],
            'test_boelter4_12': [[0, 32, 0], [33, 141, 0], [463, 519, 0], [520, 597, 0], [598, 605, 0],
                                 [1233, 1293, 0]],
            'test_boelter4_9': [[0, 221, 0], [308, 466, 0], [1215, 1270, 0]],
            'test_boelter4_4': [[0, 183, 0], [281, 529, 0], [530, 714, 0]],
            'test_boelter4_3': [[0, 117, 0]],
            'test_boelter4_1': [[0, 252, 0], [253, 729, 0], [730, 1202, 0], [1203, 1237, 0]],
            'test_boelter3_13': [],
            'test_boelter3_11': [[255, 424, 0], [599, 692, 0]],
            'test_boelter3_6': [[281, 498, 0], [499, 639, 0], [696, 748, 0], [749, 788, 0]],
            'test_boelter3_4': [[781, 817, 0]],
            'test_boelter3_0': [[0, 102, 0], [103, 480, 0], [481, 703, 0], [704, 768, 0]],
            'test_boelter2_15': [[0, 46, 0]],
            'test_boelter2_12': [[0, 163, 0], [445, 519, 0], [584, 660, 0]],
            'test_boelter2_5': [[0, 94, 0], [655, 1206, 0]],
            'test_boelter2_4': [],
            'test_boelter2_2': [[0, 145, 0], [146, 224, 0], [225, 271, 0], [272, 392, 0], [393, 454, 0],
                                [455, 762, 0], [763, 982, 0], [983, 1412, 0]],
            'test_boelter_21': [[239, 285, 0], [286, 310, 0], [374, 457, 0]],
        }
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

    def check_norm(self, features):
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
        return features

    def show_joints_id(self, skeleton):


        pass

    def extract_body_motion_features(self, skeletons):
        feature_dim = len(self.check_joints) * len(self.check_frames) * 3
        video_features = np.empty((0, feature_dim))
        for frame_id, skeleton in enumerate(skeletons):
            skeleton = np.array(skeleton)
            frame_feature = np.empty((1, 0))
            torso = skeleton[1]
            for check_frame in self.check_frames:
                if frame_id + check_frame == frame_id:
                    features = skeleton[self.check_joints] - torso
                    # features = self.check_norm(features)
                    features = features.reshape(1, -1)
                    frame_feature = np.hstack([frame_feature, features])
                else:
                    pre_frame_id = min(0, check_frame + frame_id)
                    pre_skele = skeletons[pre_frame_id]
                    features = skeleton - pre_skele
                    features = features[self.check_joints]
                    # features = self.check_norm(features)
                    features = features.reshape(1, -1)
                    frame_feature = np.hstack([frame_feature, features])
            video_features = np.vstack([video_features, frame_feature])
        return video_features

    def extract_hand_features(self, skeletons):
        feature_dim = 6 + 6 + 36
        video_feature = np.empty((0, feature_dim))
        for frame_id, skeleton in enumerate(skeletons):
            skeleton = np.array(skeleton)
            frame_feature = np.empty((1, 0))
            torso = skeleton[1]
            head = skeleton[20]
            left_hand = skeleton[11]
            right_hand = skeleton[7]
            for check_frame in range(-6, 1, 1):
                if check_frame == 0:
                    left_head = left_hand - head
                    left_head = left_head.reshape((1, -1))
                    # left_head = self.check_norm(left_head).reshape(1, -1)
                    left_torso = left_hand - torso
                    left_torso = left_torso.reshape((1, -1))
                    # left_torso = self.check_norm(left_torso).reshape(1, -1)
                    right_head = right_hand - head
                    right_head = right_head.reshape((1, -1))
                    # right_head = self.check_norm(right_head).reshape(1, -1)
                    right_torso = right_hand - torso
                    right_torso = right_torso.reshape((1, -1))
                    # right_torso = self.check_norm(right_torso).reshape(1, -1)
                    frame_feature = np.hstack([frame_feature, left_head, left_torso, right_head, right_torso])
                else:
                    pre_frame_id = min(0, check_frame + frame_id)
                    pre_skele = skeletons[pre_frame_id]
                    features = skeleton - pre_skele
                    # features = self.check_norm(features)
                    inds = [7, 11]
                    features = features[inds]
                    features = features.reshape(1, -1)
                    frame_feature = np.hstack([frame_feature, features])
            video_feature = np.vstack([video_feature, frame_feature])
        return video_feature

    def extract_foot_features(self, skeletons):
        feature_dim = 6 + 6
        video_feature = np.empty((0, feature_dim))
        for frame_id, skeleton in enumerate(skeletons):
            skeleton = np.array(skeleton)
            frame_feature = np.empty((1, 0))
            torso = skeleton[1]
            head = skeleton[20]
            left_foot = skeleton[19]
            right_foot = skeleton[15]
            left_head = left_foot - head
            left_head = left_head.reshape((1, -1))
            # left_head = self.check_norm(left_head).reshape(1, -1)
            left_torso = left_foot - torso
            left_torso = left_torso.reshape((1, -1))
            # left_torso = self.check_norm(left_torso).reshape(1, -1)
            right_head = right_foot - head
            right_head = right_head.reshape((1, -1))
            # right_head = self.check_norm(right_head).reshape(1, -1)
            right_torso = right_foot - torso
            right_torso = right_torso.reshape((1, -1))
            # right_torso = self.check_norm(right_torso).reshape(1, -1)
            frame_feature = np.hstack([frame_feature, left_head, left_torso, right_head, right_torso])
            video_feature = np.vstack([video_feature, frame_feature])
        return video_feature

    def point2screen(self, points):
        K = [607.13232421875, 0.0, 638.6468505859375, 0.0, 607.1067504882812, 367.1607360839844, 0.0, 0.0, 1.0]
        K = np.reshape(np.array(K), [3, 3])
        rot_points = np.array(points)
        rot_points = rot_points
        points_camera = rot_points.reshape(3, 1)

        project_matrix = np.array(K).reshape(3, 3)
        points_prj = project_matrix.dot(points_camera)
        points_prj = points_prj.transpose()
        if not points_prj[:, 2][0] == 0.0:
            points_prj[:, 0] = points_prj[:, 0] / points_prj[:, 2]
            points_prj[:, 1] = points_prj[:, 1] / points_prj[:, 2]
        points_screen = points_prj[:, :2]
        assert points_screen.shape == (1, 2)
        points_screen = points_screen.reshape(-1)
        return points_screen

    def extract_hog_features(self, skeletons, img_names, depths):
        feature_dim = 648
        video_feature = np.empty((0, feature_dim))
        for frame_id, skeleton in enumerate(skeletons):
            skeleton = np.array(skeleton)
            frame_feature = np.empty((1, 0))
            img_name = img_names[frame_id]
            to_draw_img = cv2.imread(img_name)
            depth_name = depths[frame_id]
            depth_img = np.load(depth_name)
            for ind, joint_id in enumerate(self.anchor_joints):
                joint = skeleton[joint_id]
                screen_joint = self.point2screen(joint)
                x, y = screen_joint[0], screen_joint[1] + self.shift[ind]
                width = self.box_size[ind]
                y_min = min(max(y - width, 0), to_draw_img.shape[0] - 100)
                y_max = max(min(y + width, to_draw_img.shape[0]), 100)
                x_min = min(max(x - width, 0), to_draw_img.shape[1] - 100)
                x_max = max(min(x + width, to_draw_img.shape[1]), 100)

                box = [x_min, y_min, x_max, y_max]
                cropped_image = to_draw_img[int(y_min):int(y_max), int(x_min):int(x_max), :]
                cropped_image = cv2.resize(cropped_image, self.hog.winSize)
                hog_feature = self.hog.compute(cropped_image).reshape(1, -1)
                frame_feature = np.hstack([frame_feature, hog_feature])
                temp = to_draw_img.copy()
                # cv2.circle(temp, (int(joint[0]), int(joint[1])), 3, (255, 0, 0), thickness=1)
                # cv2.rectangle(temp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
                # cv2.imshow('img', cropped_image)
                # cv2.waitKey(20)
                cropped_depth_img = depth_img[int(y_min):int(y_max), int(x_min):int(x_max)]
                cropped_depth_img = cropped_depth_img.astype(np.uint8)
                cropped_depth_img = cv2.resize(cropped_depth_img, self.hog.winSize)
                hog_feature = self.hog.compute(cropped_depth_img).reshape(1, -1)
                frame_feature = np.hstack([frame_feature, hog_feature])
            video_feature = np.vstack([video_feature, frame_feature])
        return video_feature

    def check_location(self, clip):
        names = clip.split('_')
        print(names)
        if len(names) == 1:
            return 'video_lib'
        else:
            name = names[1]
            if name == '9434':
                return 'video_9434'
            elif name == '94342':
                return 'video_94342'
            elif name == 'boelter':
                return 'video_bhlib'
            elif name == 'boelter2':
                return 'video_bhlib2'
            elif name == 'boelter3':
                return 'video_bhlib3'
            elif name == 'boelter4':
                return 'video_bhlib4'

    def get_orientation(self, tracker):
        a = tracker[9] - tracker[5]
        b = tracker[0] - tracker[5]
        c = np.cross(a, b)
        c_norm = np.linalg.norm(c)
        if c_norm > 0:
            c = c / c_norm
        return [c[0], c[1], 0]

    def cal_angle(self, orientation):
        theta = np.arctan2(orientation[1], orientation[0])
        return theta

    def cal_hand_relative(self, tracker_t, battery_t, frame_feature, joint_id):
        battery_shoulder = battery_t[joint_id]
        tracker_shoulder = tracker_t[joint_id]
        battery_hand = battery_t[joint_id + 2] - battery_t[joint_id]
        tracker_hand = tracker_t[joint_id + 2] - tracker_t[joint_id]
        shoulder_relative_dirct = (battery_shoulder - tracker_shoulder).reshape((1, -1))
        shoulder_relative_dirct = self.check_norm(shoulder_relative_dirct)
        tracker_hand = self.check_norm(tracker_hand)
        angle = shoulder_relative_dirct.dot(tracker_hand).reshape((1, -1))
        frame_feature = np.hstack([frame_feature, angle])
        battery_hand = self.check_norm(battery_hand)
        angle = -shoulder_relative_dirct.dot(battery_hand).reshape((1, -1))
        frame_feature = np.hstack([frame_feature, angle])
        if joint_id == 9:
            battery_right_hand = battery_t[7] - battery_t[5]
            tracker_right_hand = tracker_t[7] - tracker_t[5]
            battery_right_hand = self.check_norm(battery_right_hand)
            tracker_right_hand = self.check_norm(tracker_right_hand)
            angle1 = tracker_hand.dot(battery_hand).reshape((1, -1))
            angle2 = tracker_hand.dot(battery_right_hand).reshape((1, -1))
            angle3 = tracker_right_hand.dot(battery_hand).reshape((1, -1))
            angle4 = tracker_right_hand.dot(battery_right_hand).reshape((1, -1))
            frame_feature = np.hstack([frame_feature, angle1, angle2, angle3, angle4])
        return frame_feature


    def extract_pair_features(self, trackers, batteries, battery_gazes, tracker_gazes):
        video_feature = np.empty((0, len(self.check_joints)*3 + 7 + 1 + 4))
        for frame_id, tracker in enumerate(trackers):
            battery = batteries[frame_id]
            tracker = np.array(tracker)
            battery = np.array(battery)
            # transform
            tracker_center = np.mean(tracker, axis=0)
            orientation = self.get_orientation(tracker)
            angle = self.cal_angle(orientation)
            tracker_t = tracker - tracker_center
            r = R.from_euler('z', angle)
            tracker_t = r.apply(tracker_t)

            battery_t = battery - tracker_center
            battery_center = np.mean(battery_t, axis=0)
            battery_temp = battery_t - battery_center
            battery_temp = r.apply(battery_temp)
            battery_t = battery_temp + battery_center

            frame_feature = np.empty((1, 0))

            # relative joint
            dirct = battery_t[self.check_joints] - tracker_t[self.check_joints]
            dirct = dirct.reshape((1, -1))
            dirct = self.check_norm(dirct)
            frame_feature = np.hstack([frame_feature, dirct])

            # relative gaze
            if battery_gazes[frame_id]:
                battery_gaze = battery_gazes[frame_id][0] - battery_gazes[frame_id][1]
            else:
                battery_gaze = np.array([0, 0, 0])
            if tracker_gazes[frame_id]:
                tracker_gaze = tracker_gazes[frame_id][0] - tracker_gazes[frame_id][1]
            else:
                tracker_gaze = np.array([0, 0, 0])
            battery_gaze = self.check_norm(battery_gaze)
            tracker_gaze = self.check_norm(tracker_gaze)
            battery_gaze_t = r.apply(battery_gaze)
            tracker_gaze_t = r.apply(tracker_gaze)

            battery_head = battery_t[20]
            tracker_head = tracker_t[20]
            head_relative_dirct = battery_head - tracker_head
            head_relative_dirct = self.check_norm(head_relative_dirct)
            angle = head_relative_dirct.dot(tracker_gaze_t).reshape((1, -1))
            frame_feature = np.hstack([frame_feature, angle])
            angle = -head_relative_dirct.dot(battery_gaze_t).reshape((1, -1))
            frame_feature = np.hstack([frame_feature, angle])
            angle = tracker_gaze_t.dot(battery_gaze_t).reshape((1, -1))
            frame_feature = np.hstack([frame_feature, angle])
            # relative dist
            dist = np.linalg.norm(battery_center - tracker_center).reshape((1, -1))
            frame_feature = np.hstack([frame_feature, dist])

            # left hand feature
            frame_feature = self.cal_hand_relative(tracker_t, battery_t, frame_feature, 9)

            # right hand feature
            frame_feature = self.cal_hand_relative(tracker_t, battery_t, frame_feature, 5)

            video_feature = np.vstack([video_feature, frame_feature])
        return video_feature

    def extract_head_motion_features(self, skeletons):
        feature_dim = 3*3 + 4*6*3
        video_feature = np.empty((0, feature_dim))
        for frame_id, skeleton in enumerate(skeletons):
            skeleton = np.array(skeleton)
            frame_feature = np.empty((1, 0))
            head = skeleton[20]
            for check_frame in range(-6, 1, 1):
                if check_frame == 0:
                    direct = skeleton[self.check_head_joints_no_chin] - head
                    # direct = self.check_norm(direct).reshape((1, -1))
                    direct = direct.reshape((1, -1))
                    frame_feature = np.hstack([frame_feature, direct])
                else:
                    pre_frame_id = min(0, check_frame + frame_id)
                    pre_skele = skeletons[pre_frame_id]
                    features = skeleton - pre_skele
                    features = features[self.check_head_joints]
                    # features = self.check_norm(features)
                    features = features.reshape(1, -1)
                    frame_feature = np.hstack([frame_feature, features])
            video_feature = np.vstack([video_feature, frame_feature])
        return video_feature

    def check_gaze(self, gaze):
        if gaze:
            gaze_t = gaze[0] - gaze[1]
        else:
            gaze_t = np.array([0, 0, 0])
        return gaze_t

    def extract_gaze_motion_features(self, gazes):
        feature_dim = 3*7
        video_feature = np.empty((0, feature_dim))
        for frame_id in range(len(gazes)):
            frame_feature = np.empty(((1, 0)))
            gaze = gazes[frame_id]
            gaze = self.check_gaze(gaze)
            gaze_t = self.check_norm(gaze).reshape((1, -1))
            frame_feature = np.hstack([frame_feature, gaze_t])
            for check_frame in range(-6, 0, 1):
                pre_frame_id = min(0, check_frame + frame_id)
                pre_gaze = gazes[pre_frame_id]
                pre_gaze = self.check_gaze(pre_gaze).reshape((1, -1))
                # pre_gaze = self.check_norm(pre_gaze).reshape((1, -1))
                features = gaze_t - pre_gaze
                # features = self.check_norm(features)
                frame_feature = np.hstack([frame_feature, features])
            video_feature = np.vstack([video_feature, frame_feature])
        return video_feature

    def extract_obj_features(self, skeletons, gazes, cate_file, mask_names):
        feature_dim = 30
        video_feature = np.empty((0, feature_dim))
        for frame_id, skeleton in enumerate(skeletons):
            frame_features = np.empty((1, 0))
            skeleton = np.array(skeleton)
            torso = skeleton[20]
            mask_name = mask_names[frame_id]
            with open(mask_name, 'rb') as f:
                masks = joblib.load(f)
            with open(cate_file, 'rb') as f:
                category = joblib.load(f)
            distance = np.empty((0, 4))
            for obj_name, frame_cates in category.items():
                frame_cate = frame_cates[frame_id]
                if frame_cate[0]:
                    cate_id = frame_cate[1]
                    sub_id = frame_cate[2]
                    mask = masks[cate_id][1][sub_id]
                    avg_col = np.array(mask).mean(axis=1)
                    if np.array(mask)[avg_col != 0, :].shape[0] == 0:
                        distance = np.vstack([distance, np.array([0, 0, 0, 100])])
                    else:
                        obj_center = np.array(mask)[avg_col != 0, :].mean(axis=0)
                        dist = np.linalg.norm(obj_center - torso)
                        obj_center = np.hstack([obj_center, dist])
                        distance = np.vstack([distance, obj_center])
                else:
                    distance = np.vstack([distance, np.array([0, 0, 0, 100])])

            idx = np.argsort(distance[:, 3])[:10]
            top_objs = distance[idx][:, :3]

            left_hand = skeleton[7]
            right_hand = skeleton[11]
            gaze = gazes[frame_id]
            gaze = self.check_gaze(gaze).reshape((-1, 1))
            gaze = self.check_norm(gaze)

            left_dist = top_objs - left_hand
            left_dist = np.linalg.norm(left_dist, axis=1).reshape((1, -1))
            right_dist = top_objs - right_hand
            right_dist = np.linalg.norm(right_dist, axis=1).reshape((1, -1))
            eyes_related_id = [21, 22, 24]
            eyes_related = skeleton[eyes_related_id]
            eye_center = np.mean(eyes_related, axis=0)

            avg_col = np.array(top_objs).mean(axis=1)
            top_objs[avg_col != 0] = top_objs[avg_col != 0] - eye_center
            dirct_norm = np.linalg.norm(top_objs, axis=1)
            idx = np.where(dirct_norm > 0)[0]
            if len(idx) > 0:
                top_objs = top_objs[dirct_norm > 0]/dirct_norm[dirct_norm > 0].reshape((-1, 1))
            gaze_angle = top_objs.dot(gaze).reshape((1, -1))
            frame_features = np.hstack([frame_features, left_dist, right_dist, gaze_angle])
            video_feature = np.vstack([video_feature, frame_features])
        return video_feature


    def load_file(self):
        data_path = '/home/shuwen/data/data_preprocessing2/post_skeletons/'
        img_path = '/home/shuwen/data/data_preprocessing2/annotations'
        depth_path = '/home/shuwen/data/data_preprocessing2/post_images/kinect'
        save_folder = '/home/shuwen/data/data_preprocessing2/skeleton_feature_no_norm/'
        battery_gaze_path = '/home/shuwen/data/data_preprocessing2/merge2gaze/'
        tracker_gaze_path = '/home/shuwen/data/data_preprocessing2/merge2gaze_tracker2/'
        mask_path = '/home/shuwen/Downloads/pointclouds/'
        cate_path = '/home/shuwen/data/data_preprocessing2/track_cate_with_frame/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # clips = os.listdir(data_path)
        clips = self.event_seg_battery.keys()
        clips = ['test_boelter2_2']
        for clip in clips:
            # if os.path.exists(save_folder + clip + '.p'):
            #    continue
            print('*******{}*******'.format(clip))
            location = self.check_location(clip)
            print(location)
            depths = sorted(glob.glob(os.path.join(depth_path, location, clip, '*.npy')))
            img_names = sorted(glob.glob(os.path.join(img_path, clip, 'kinect', '*jpg')))
            cate_file = os.path.join(cate_path, clip, clip + '.p')
            mask_names = sorted(glob.glob(os.path.join(mask_path, clip, '*.p')))
            print(os.path.join(img_path, clip, 'kinect', '*.jpg'))
            features = {}
            with open(os.path.join(data_path, clip, 'skele1.p')) as f:
                skeletons1 = pickle.load(f)
            with open(os.path.join(data_path, clip, 'skele2.p')) as f:
                skeletons2 = pickle.load(f)
            if self.trackers[clip] == 'skele1.p':
                trackers = skeletons1
                battery = skeletons2
            else:
                trackers = skeletons2
                battery = skeletons1

            with open(os.path.join(battery_gaze_path, clip, 'target.p'), 'rb') as f:
                battery_gaze = joblib.load(f)
            with open(os.path.join(tracker_gaze_path, clip, 'target.p'), 'rb') as f:
                tracker_gaze = joblib.load(f)

            # pair_features = self.extract_pair_features(trackers, battery, battery_gaze, tracker_gaze)
            total = {'tracker': [trackers, tracker_gaze], 'battery': [battery, battery_gaze]}
            for i in [1, 2]:
                if str(i) == self.trackers[clip].split('.')[0][-1]:
                    print('tracker')
                    skeletons, gaze = total['tracker']
                else:
                    print('battery')
                    skeletons, gaze = total['battery']
                # head features
                head_features = self.extract_head_motion_features(skeletons)
                gaze_features = self.extract_gaze_motion_features(gaze)
                #skeletons features
                body_motion_features = self.extract_body_motion_features(skeletons)
                hand_features = self.extract_hand_features(skeletons)
                foot_features = self.extract_foot_features(skeletons)
                #object features
                object_feature = self.extract_obj_features(skeletons, gaze, cate_file, mask_names)
                # print(object_feature.shape)
                #visual features
                # hog_features = self.extract_hog_features(skeletons, img_names, depths)
                # print(body_motion_features.shape[1] + hand_features.shape[1] + foot_features.shape[1], hog_features.shape[1], pair_features.shape[1])
                # total_feature = np.hstack([body_motion_features, hand_features, foot_features, hog_features, pair_features])
                total_feature = np.hstack([head_features, gaze_features, body_motion_features, hand_features, foot_features, object_feature])
                mean = total_feature.mean()
                std = total_feature.std()
                total_feature = (total_feature - mean)/std
                # print(head_features.shape[1], gaze_features.shape[1], body_motion_features.shape[1], hand_features.shape[1], foot_features.shape[1], object_feature.shape[1])
                # total_feature = object_feature
                features[i] = total_feature
                # print(total_feature.shape)
            with open(save_folder + clip + '.p', 'wb') as f:
                pickle.dump(features, f)


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.load_file()