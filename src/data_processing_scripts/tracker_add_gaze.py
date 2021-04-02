import glob
import cv2
import pickle
import numpy as np
import os
import shutil 

data_path = './post_images/tracker/'
video_folders = os.listdir(data_path)
for video_folder in video_folders:
    clips = os.listdir(data_path + video_folder)
    print(video_folder)
    for clip in clips:
        print(clip)
        img_names = sorted(glob.glob(data_path + video_folder + '/' + clip + '/*.jpg'))
        with open(data_path + video_folder + '/' + clip + '/' + clip + '.p') as f:
            gazes = pickle.load(f)
        assert len(gazes) == len(img_names)
        save_path = './annotations/' + clip + '/tracker/'
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # else:
        #     continue

        for frame_id, img_name in enumerate(img_names):
            img = cv2.imread(img_name)
            gaze = gazes[frame_id]
            save_img_name = img_name.split('/')[-1]
            if not np.isnan(gaze['gaze'][0]) and not np.isnan(gaze['gaze'][1]):
                cv2.circle(img, (int(gaze['gaze'][0]), int(gaze['gaze'][1])), 7, (255, 0, 255), thickness=5)
                cv2.imwrite(save_path + save_img_name, img)
            else:
                cv2.imwrite(save_path + save_img_name, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(20)