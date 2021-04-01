import os
import joblib
import numpy as np
import pickle
import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GazeSmooth:
    def __init__(self):
        self.check_frames = [-3, -2, -1, 1, 2, 3]
        self.weight = 0.7
        self.eye_frames = [21, 22, 24]
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

    def check_norm(self, vector):
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 0:
            vector = vector/vector_norm
        return vector

    def pre_smooth(self):
        battery_gaze_path = './transformed_new_results'
        tracker_gaze_path = './transformed_new_results_tracker'
        paths = {'battery':battery_gaze_path, 'tracker':tracker_gaze_path}
        save_path = '/home/shuwen/data/data_preprocessing2/pre_gaze_smooth_360'
        for key, path in paths.items():
            save_path_prefix = save_path + '_' + key
            if not os.path.exists(save_path_prefix):
                os.makedirs(save_path_prefix)
            clips = os.listdir(path)
            for clip in clips:
                print(clip)
                gaze_file = os.path.join(path, clip, 'target.p')
                with open(gaze_file, 'rb') as f:
                    gazes = joblib.load(gaze_file)
                new_gazes = []
                temp_id = None
                for gaze_id, gaze in enumerate(gazes):
                    if gaze:
                        gaze_t = gaze[0] - gaze[1]
                        gaze_t = self.check_norm(gaze_t)
                        new_gazes.append(gaze_t)
                    else:
                        new_gazes.append(np.array([0, 0, 0]))
                    if not gaze:
                        if temp_id == None:
                            temp_id = gaze_id
                        continue
                    elif not temp_id == None:
                        if gaze_id - temp_id > 0 and gaze_id - temp_id < 15:
                            for j in range(temp_id, gaze_id):
                                gaze_t = (new_gazes[gaze_id] - new_gazes[temp_id - 1]) / (
                                            gaze_id - temp_id + 1) * (
                                                       j - temp_id + 1) + new_gazes[temp_id - 1]
                                gaze_t = self.check_norm(gaze_t)
                                new_gazes[j] = gaze_t
                        temp_id = None

                with open(os.path.join(save_path_prefix, clip + '.p'), 'wb') as f:
                    pickle.dump(new_gazes, f)

    def pre_smooth_360(self):
        battery_gaze_path = '../3d_pose2gaze/transformed_normal_new_results'
        tracker_gaze_path = '../3d_pose2gaze/transformed_normal_new_results_tracker'
        paths = {'battery':battery_gaze_path, 'tracker':tracker_gaze_path}
        save_path = '/home/shuwen/data/data_preprocessing2/pre_gaze_smooth_360'
        for key, path in paths.items():
            save_path_prefix = save_path + '_' + key
            if not os.path.exists(save_path_prefix):
                os.makedirs(save_path_prefix)
            clips = os.listdir(path)
            for clip in clips:
                img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
                img_names = sorted(glob.glob(os.path.join(img_path, clip.split('.')[0], 'kinect', '*.jpg')))
                if key == 'battery':
                    with open('../3d_pose2gaze/record_bbox/' + clip , 'rb') as f:
                        tracking_id = joblib.load(f)
                else:
                    with open('../3d_pose2gaze/tracker_record_bbox/' + clip , 'rb') as f:
                        tracking_id = joblib.load(f)
                print(clip)
                gaze_file = os.path.join(path, clip )
                with open(gaze_file, 'rb') as f:
                    gazes = joblib.load(gaze_file)
                new_gazes = []
                temp_id = None
                for gaze_id in range(len(img_names)):
                    if gaze_id in gazes:
                        bbox, eyes, gaze_center = tracking_id[gaze_id][0]
                        gaze = gazes[gaze_id][0]
                        gaze = np.array(gaze)
                        gaze_t = gaze - gaze_center
                        gaze_t = self.check_norm(gaze_t)
                        new_gazes.append(gaze_t)
                    else:
                        new_gazes.append(np.array([0, 0, 0]))
                    if not gaze_id in gazes:
                        if temp_id == None:
                            temp_id = gaze_id
                        continue
                    elif not temp_id == None:
                        if gaze_id - temp_id > 0 and gaze_id - temp_id < 60:
                            for j in range(temp_id, gaze_id):
                                gaze_t = (new_gazes[gaze_id] - new_gazes[temp_id - 1]) / (
                                            gaze_id - temp_id + 1) * (
                                                       j - temp_id + 1) + new_gazes[temp_id - 1]
                                gaze_t = self.check_norm(gaze_t)
                                new_gazes[j] = gaze_t
                        temp_id = None
                if temp_id:
                    if len(img_names) - temp_id > 0 and len(img_names) - temp_id < 60:
                        for j in range(temp_id, len(img_names)):
                            gaze_t = new_gazes[temp_id - 1]
                            gaze_t = self.check_norm(gaze_t)
                            new_gazes[j] = gaze_t
                with open(os.path.join(save_path_prefix, clip), 'wb') as f:
                    pickle.dump(new_gazes, f)

    def smooth(self):
        battery_gaze_path = '/home/shuwen/data/data_preprocessing2/pre_gaze_smooth_360_battery/'
        tracker_gaze_path = '/home/shuwen/data/data_preprocessing2/pre_gaze_smooth_360_tracker/'
        paths = {'battery': battery_gaze_path, 'tracker': tracker_gaze_path}
        save_path = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360'
        for key, path in paths.items():
            save_path_prefix = save_path + '_' + key
            if not os.path.exists(save_path_prefix):
                os.makedirs(save_path_prefix)
            clips = sorted(glob.glob(os.path.join(path, '*.p')))
            for clip in clips:
                print(clip)
                with open(clip, 'rb') as f:
                    gazes = pickle.load(f)
                print(len(gazes))
                new_gazes = []
                for gaze_id, gaze in enumerate(gazes):
                    neibor_gazes = np.empty((0, 3))
                    for check_frame in self.check_frames:
                        if check_frame < 0:
                            pre_id = max(0, gaze_id + check_frame)
                        else:
                            pre_id = min(len(gazes) - 1, gaze_id + check_frame)
                        if gazes[pre_id].mean() != 0:
                            neibor_gazes = np.vstack([neibor_gazes, gazes[pre_id]])
                    if neibor_gazes.shape[0] == 0:
                        avg_gaze = np.array([0, 0, 0])
                    else:
                        avg_gaze = neibor_gazes.mean(axis=0)
                    # if gaze_id == 0 and clip.split('/')[-1] == 'test_94342_1.p':
                    #     print(gaze, avg_gaze, neibor_gazes)
                    # if gaze_id == 1 and clip.split('/')[-1] == 'test_94342_1.p':
                    #     print(gaze, avg_gaze, neibor_gazes)
                    gaze = self.weight * gaze + (1 - self.weight) * avg_gaze
                    new_gazes.append(gaze)
                with open(os.path.join(save_path_prefix, clip.split('/')[-1]), 'wb') as f:
                    pickle.dump(new_gazes, f)

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

    def visualize_360(self, clip):
        img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        img_names = sorted(glob.glob(os.path.join(img_path, clip, 'kinect', '*.jpg')))
        print(os.path.join(img_path, clip, 'kinect', '*.jpg'))
        skeleton_path = '/home/shuwen/data/data_preprocessing2/post_skeletons/' + clip
        battery_gaze_file = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_battery/' + clip + '.p'
        tracker_gaze_path = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_tracker/' + clip + '.p'
        save_path = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_vis/' + clip + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(skeleton_path + '/' + 'skele1.p', 'rb') as f:
            skeletons1 = pickle.load(f)
        with open(skeleton_path + '/' + 'skele2.p', 'rb') as f:
            skeletons2 = pickle.load(f)
        skeletons = {1: skeletons1, 2: skeletons2}
        with open(battery_gaze_file, 'rb') as f:
            battery_gaze = pickle.load(f)
        with open(tracker_gaze_path, 'rb') as f:
            tracker_gaze = pickle.load(f)
        gazes = {'battery': battery_gaze, 'tracker': tracker_gaze}
        with open('../3d_pose2gaze/tracker_record_bbox/' + clip + '.p', 'rb') as f:
            tracker_tracking_id = joblib.load(f)
        with open('../3d_pose2gaze/record_bbox/' + clip + '.p', 'rb') as f:
            battery_tracking_id = joblib.load(f)
        for frame_id, img_name in enumerate(img_names):
            img = cv2.imread(img_name)
            # raw_input('Enter')
            for i in range(1, 3):
                if self.trackers[clip].split('.')[0][-1] == str(i):
                    gaze_i = gazes['tracker'][frame_id]
                    tracking_id = tracker_tracking_id
                else:
                    gaze_i = gazes['battery'][frame_id]
                    tracking_id = battery_tracking_id
                if frame_id in tracking_id:
                    bbox, eyes, eye_center = tracking_id[frame_id][0]
                else:
                    skeleton = np.array(skeletons[i][frame_id])
                    eye_center = skeleton.mean(axis=0)
                eye_screen = self.point2screen(eye_center)
                gaze_screen = self.point2screen(eye_center + gaze_i)
                print(eye_screen, gaze_screen)
                img = cv2.line(img, (int(gaze_screen[0]), int(gaze_screen[1])),
                               (int(eye_screen[0]), int(eye_screen[1])), color=(255, 0, 0), thickness=3)
                save_name = save_path + '{0:04}'.format(frame_id) + '.jpg'
                # cv2.imshow('img', img)
                # cv2.waitKey(20)
                cv2.imwrite(save_name, img)

    def visualize(self, clip):
        img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        img_names = sorted(glob.glob(os.path.join(img_path, clip, 'kinect', '*.jpg')))
        print(os.path.join(img_path, clip, 'kinect', '*.jpg'))
        skeleton_path = '/home/shuwen/data/data_preprocessing2/post_skeletons/' + clip
        battery_gaze_file = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_battery/' + clip + '.p'
        tracker_gaze_path = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_tracker/' + clip + '.p'
        save_path = '/home/shuwen/data/data_preprocessing2/gaze_smooth_360_vis/' + clip + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(skeleton_path + '/' + 'skele1.p', 'rb') as f:
            skeletons1 = pickle.load(f)
        with open(skeleton_path + '/' + 'skele2.p', 'rb') as f:
            skeletons2 = pickle.load(f)
        skeletons = {1: skeletons1, 2: skeletons2}
        with open(battery_gaze_file, 'rb') as f:
            battery_gaze = pickle.load(f)
        with open(tracker_gaze_path, 'rb') as f:
            tracker_gaze = pickle.load(f)
        gazes = {'battery': battery_gaze, 'tracker': tracker_gaze}
        for frame_id, img_name in enumerate(img_names):
            img = cv2.imread(img_name)
            # raw_input('Enter')
            for i in range(1, 3):
                if self.trackers[clip].split('.')[0][-1] == str(i):
                    gaze_i = gazes['tracker'][frame_id]
                else:
                    gaze_i = gazes['battery'][frame_id]
                skeleton = np.array(skeletons[i][frame_id])
                eye_center = skeleton.mean(axis=0)
                eye_screen = self.point2screen(eye_center)
                gaze_screen = self.point2screen(eye_center + gaze_i)

                print(eye_screen, gaze_screen)
                img = cv2.line(img, (int(gaze_screen[0]), int(gaze_screen[1])),
                               (int(eye_screen[0]), int(eye_screen[1])), color=(255, 0, 0), thickness = 3)
                save_name = save_path + '{0:04}'.format(frame_id) + '.jpg'
                # cv2.imshow('img', img)
                # cv2.waitKey(20)
                cv2.imwrite(save_name, img)

    def extract_gaze(self, skeleton):
        left_eye = skeleton[22]
        right_eye = skeleton[24]
        chin = skeleton[20]
        a = left_eye - right_eye
        b = chin - right_eye
        normal = np.cross(b, a)
        normal = self.check_norm(normal)
        return normal

    def extract_skeleton_gaze(self):
        skeleton_path = './post_skeletons/'
        clips = os.listdir(skeleton_path)
        battery_save_path = './estimated_gazes_battery/'
        tracker_save_path = './estimated_gazes_tracker/'
        if not os.path.exists(battery_save_path):
            os.makedirs(battery_save_path)
        if not os.path.exists(tracker_save_path):
            os.makedirs(tracker_save_path)
        for clip in clips:
            for i in range(1, 3):
                print(clip, i)
                gazes = []
                if self.trackers[clip].split('.')[0][-1] == str(i):
                    save_file = tracker_save_path + clip + '.p'
                else:
                    save_file = battery_save_path + clip + '.p'
                skeleton_file = os.path.join(skeleton_path, clip, 'skele{}.p'.format(i))
                with open(skeleton_file, 'rb') as f:
                    skeletons = pickle.load(f)
                skeletons = np.array(skeletons)
                for frame_id, skeleton in enumerate(skeletons):
                    gaze_estimate = self.extract_gaze(skeleton)
                    gazes.append(gaze_estimate)
                with open(save_file, 'wb') as f:
                    pickle.dump(gazes, f)

    def smooth_gaze_center(self):
        data_path = './post_images/tracker/'
        video_folders = os.listdir(data_path)
        save_path = './tracker_gt_smooth/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for location in video_folders:
            clips = os.listdir(data_path + location)
            for clip in clips:
                print(clip)
                gaze_path = os.path.join(data_path, location, clip, clip + '.p')
                img_names = sorted(glob.glob(os.path.join(data_path, location, clip, '*.jpg')))
                with open(gaze_path, 'rb') as f:
                    gazes = joblib.load(f)
                new_gazes = []
                temp_id = None
                for frame_id in range(len(img_names)):
                    gaze = gazes[frame_id]['gaze']
                    if np.isnan(gaze[0]) or np.isnan(gaze[1]):
                        new_gazes.append(np.array([0, 0]))
                    else:
                        new_gazes.append(np.array(gaze))
                    if np.isnan(gaze[0]) or np.isnan(gaze[1]):
                        if temp_id == None:
                            temp_id = frame_id
                        continue
                    elif not temp_id == None:
                        for j in range(temp_id, frame_id):
                            gaze_t = (new_gazes[frame_id] - new_gazes[temp_id - 1]) / (
                                    frame_id - temp_id + 1) * (
                                             j - temp_id + 1) + new_gazes[temp_id - 1]
                            new_gazes[j] = gaze_t
                        temp_id = None
                with open(os.path.join(save_path, clip + '.p'), 'wb') as f:
                    pickle.dump(new_gazes, f)

    def visualize_gaze(self):
        clip = 'test_boelter2_15'
        with open('./tracker_gt_smooth/test_boelter2_15.p', 'rb') as f:
            gazes = pickle.load(f)
        img_names = sorted(glob.glob('./annotations/' + clip + '/tracker/*.jpg'))
        for frame_id, img_name in enumerate(img_names):
            img = cv2.imread(img_name)
            gaze = gazes[frame_id]
            print(gaze)
            cv2.circle(img, (int(gaze[0]), int(gaze[1])), color=(0, 0, 255), radius = 3, thickness=2)
            cv2.imshow('img', img)
            cv2.waitKey(20)


if __name__ == '__main__':
    gaze_smoother = GazeSmooth()
    # interpolate for missing frames
    gaze_smoother.pre_smooth_360()
    # average gaze direction between frames
    gaze_smoother.smooth()
    # visualize
    gaze_smoother.visualize_360('test_94342_1')
    # save the smoothed gaze
    gaze_smoother.extract_skeleton_gaze()
    # gaze_smoother.smooth_gaze_center()
    gaze_smoother.visualize_gaze()








