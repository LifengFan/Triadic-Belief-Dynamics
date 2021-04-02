#!/usr/bin/env python
import sys
import cv2
import numpy as np
import os
from sys import platform
import argparse
import pickle
import glob

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
params["hand_detector"] = 2
params["body"] = 0
#params["num_gpu_start"] = 1

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


class HandRecog:
    def __init__(self, box_size = 80, face_size = 50):
        self.box_size = box_size
        self.face_size = face_size

    def point2screen(self, points, K):
        rot_points = np.array(points)
        points_camera = rot_points.reshape(3,1)
        project_matrix = np.array(K).reshape(3,3)
        points_prj = project_matrix.dot(points_camera)
        points_prj = points_prj.transpose()
        if not points_prj[:, 2][0] == 0.0:
            points_prj[:,0] = points_prj[:,0]/points_prj[:,2]
            points_prj[:,1] = points_prj[:,1]/points_prj[:,2]
        points_screen = points_prj[:,:2]
        assert points_screen.shape==(1,2)
        points_screen = points_screen.reshape(-1)
        return points_screen


    def hand_recog(self, boxes, img, K):
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
                                    (int(points_screen[0])+self.box_size+50, int(points_screen[1])+self.box_size), (255,0,0), 2)
                        rectangle.append(op.Rectangle(points_screen[0]-self.box_size, points_screen[1]-self.box_size, self.box_size*2, self.box_size*2))
            else:
                rectangle.append(op.Rectangle(0.0, 0.0, 0.0, 0.0))
                rectangle.append(op.Rectangle(0.0, 0.0, 0.0, 0.0))

            assert len(rectangle) == 2
            rectangles.append(rectangle)
        assert len(rectangles) == 2      

        cv2.imshow("hand", img_draw)
        cv2.waitKey(30)
        return rectangles
            
class HandPublisher:
    def hand2marker(self, keypoints, color, depth, K):
        fx_d, fy_d, cx_d, cy_d = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        hand_points = []
        
        for hand_id, hand in enumerate(keypoints): 
            for pid in range(hand.shape[0]):
                hand_point = []
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
                    else:
                        x,y,z = 0,0,0
                    hand_point.append([x, y, z])
                hand_points.append(hand_point)
        assert len(hand_points) == 4
        return hand_points

""" Main """
if __name__ == '__main__':
    K = [607.13232421875, 0.0, 638.6468505859375, 0.0, 607.1067504882812, 367.1607360839844, 0.0, 0.0, 1.0]
    K = np.reshape(np.array(K), [3, 3])

    recognizer = HandRecog()
    pub = HandPublisher()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    img_path = '/home/shuwen/data/data_preprocessing/post_images/kinect/'
    save_prefix = '/home/shuwen/data/data_preprocessing/posthand_images/'
    img_folders = os.listdir(img_path)
    for img_folder in img_folders:
        print('**********' + img_path + img_folder + '************')
        clips = glob.glob(img_path + img_folder + '/*')
        for clip in clips:
            print('************' + clip + '****************')
            color_img_names = sorted(glob.glob(clip + '/color*'))
            depth_img_names = sorted(glob.glob(clip + '/depth*'))
            f = open(clip + '/skele1.p', 'rb')
            skele1 = pickle.load(f)
            f = open(clip + '/skele2.p', 'rb')
            skele2 = pickle.load(f)
            hand_pointss = []
            save_path = save_prefix + img_folder + '/' + clip.split('/')[-1] 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for idx, color_img_name in enumerate(color_img_names):
                print(color_img_name)
                color = cv2.imread(color_img_name)
                depth = np.load(depth_img_names[idx])
                
                points1 = skele1[idx]
                points2 = skele2[idx]

                hand_box1 = [list(points1[7, :]), list(points1[11, :])]
                hand_box2 = [list(points2[7, :]), list(points2[11, :])]
                boxes = [hand_box1, hand_box2]
                rectangles = recognizer.hand_recog(boxes, color, K)

                datum = op.Datum()
                datum.cvInputData = color
                datum.handRectangles = rectangles
                # Process and display image
                opWrapper.emplaceAndPop([datum])
                hand_points = pub.hand2marker(datum.handKeypoints, color, depth, K)
                hand_pointss.append(hand_points)
                cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
                cv2.imwrite(save_path + '/' + '{0:04}'.format(idx) + '.jpg', datum.cvOutputData)
                cv2.waitKey(100)
            f = open(save_path + '/' +'hands.p', 'wb')
            pickle.dump(hand_pointss, f)
