import os
import pandas as pd
import pickle
import cv2
import glob
import numpy as np
import random

def get_data():
    annotation_path = './regenerate_annotation/'
    bbox_path = './reformat_annotation/'
    feature_path = './feature_single/'
    color_img_path = './annotations/'
    save_path = './attention_training/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open('./person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(annotation_path)

    for clip in clips:
        print(clip)
        training_input = []
        training_output = []
        negative_ids = []
        with open(os.path.join(annotation_path, clip), 'rb') as f:
            obj_records = pickle.load(f)
        annt = pd.read_csv(bbox_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        with open(feature_path + clip, 'rb') as f:
            features = pickle.load(f)

        if person_ids[clip.split('.')[0]] == 'P1':
            p1_hog = features[1]
            p2_hog = features[2]
        else:
            p1_hog = features[2]
            p2_hog = features[1]

        obj_names = obj_records.keys()

        for obj_name in obj_names:
            flag_m1 = 1
            flag_m2 = 1
            negative_p1 = None
            negative_p2 = None
            for frame_id in range(len(obj_records[obj_name])):
                label = np.array([0, 0, 0])

                if obj_records[obj_name][frame_id]['m1']['fluent'] == 0:
                    if flag_m1:
                        negative_p1 = frame_id
                        flag_m1 = 0
                    # img = cv2.imread(img_names[frame_id])
                    # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                    #
                    # x_min = obj_frame['x_min'].item()
                    # y_min = obj_frame['y_min'].item()
                    # x_max = obj_frame['x_max'].item()
                    # y_max = obj_frame['y_max'].item()
                    #
                    # img_patch = img[y_min:y_max, x_min:x_max]
                    #
                    # # hog
                    # hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
                    # training_input.append([img_patch, hog])
                    label[1] = 1

                if obj_records[obj_name][frame_id]['m2']['fluent'] == 0:
                    if flag_m2:
                        negative_p2 = frame_id
                        flag_m2 = 0
                    # img = cv2.imread(img_names[frame_id])
                    # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                    #
                    # x_min = obj_frame['x_min'].item()
                    # y_min = obj_frame['y_min'].item()
                    # x_max = obj_frame['x_max'].item()
                    # y_max = obj_frame['y_max'].item()
                    #
                    # img_patch = img[y_min:y_max, x_min:x_max]
                    #
                    # # hog
                    # hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
                    # training_input.append([img_patch, hog])
                    # training_output.append(2)
                    label[2] = 1

                if obj_records[obj_name][frame_id]['m1']['fluent'] == 2:
                    # img = cv2.imread(img_names[frame_id])
                    # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                    #
                    # x_min = obj_frame['x_min'].item()
                    # y_min = obj_frame['y_min'].item()
                    # x_max = obj_frame['x_max'].item()
                    # y_max = obj_frame['y_max'].item()
                    #
                    # img_patch = img[y_min:y_max, x_min:x_max]
                    #
                    # # hog
                    # hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
                    # training_input.append([img_patch, hog])
                    # training_output.append(1)
                    label[1] = 1

                if obj_records[obj_name][frame_id]['m2']['fluent'] == 2:
                    # img = cv2.imread(img_names[frame_id])
                    # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                    #
                    # x_min = obj_frame['x_min'].item()
                    # y_min = obj_frame['y_min'].item()
                    # x_max = obj_frame['x_max'].item()
                    # y_max = obj_frame['y_max'].item()
                    #
                    # img_patch = img[y_min:y_max, x_min:x_max]
                    #
                    # # hog
                    # hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
                    # training_input.append([img_patch, hog])
                    # training_output.append(2)
                    label[2] = 1

                if np.all(label == 0):
                    min_id = None
                    if negative_p1:
                        min_id = negative_p1
                    if negative_p2:
                        if min_id > negative_p2:
                            min_id = negative_p2
                    if min_id:
                        for i in range(min_id):
                            negative_ids.append([obj_name, frame_id])
                else:
                    img = cv2.imread(img_names[frame_id])
                    obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]

                    x_min = obj_frame['x_min'].item()
                    y_min = obj_frame['y_min'].item()
                    x_max = obj_frame['x_max'].item()
                    y_max = obj_frame['y_max'].item()

                    img_patch = img[y_min:y_max, x_min:x_max]

                    # hog
                    hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
                    training_input.append([img_patch, hog])
                    training_output.append(label)

        random.shuffle(negative_ids)
        ratio = len(training_input)/2
        negative_ids_picked = negative_ids[:ratio]
        for obj_name, frame_id in negative_ids_picked:
            img = cv2.imread(img_names[frame_id])
            obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]

            x_min = obj_frame['x_min'].item()
            y_min = obj_frame['y_min'].item()
            x_max = obj_frame['x_max'].item()
            y_max = obj_frame['y_max'].item()

            img_patch = img[y_min:y_max, x_min:x_max]
            # cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), thickness=2)
            # cv2.imshow('img', img)
            # cv2.waitKey(2000)

            # hog
            hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])

            training_input.append([img_patch, hog])
            training_output.append(np.array([1, 0, 0]))

        with open(save_path + clip, 'wb') as f:
            pickle.dump([training_input, training_output], f)


get_data()