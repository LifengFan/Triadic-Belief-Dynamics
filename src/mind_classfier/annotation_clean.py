import pandas as pd
import os
import glob
import joblib
import cv2
import pickle
import numpy as np

def reframe_annotation():
    #annotation_path = '/home/shuwen/Downloads/all/'
    annotation_path = '/home/lfan/Dropbox/Projects/ECCV20/annot/all/'
    #save_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    save_path='/home/lfan/Dropbox/Projects/ECCV20/reformat_annotation/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tasks = glob.glob(annotation_path + '*.txt')
    id_map = pd.read_csv('id_map.csv')
    for task in tasks:
        if not task.split('/')[-1].split('_')[2] == '1.txt':
            continue
        with open(task, 'r') as f:
            lines = f.readlines()
        task_id = int(task.split('/')[-1].split('_')[1]) + 1
        clip = id_map.loc[id_map['ID'] == task_id].folder
        print(task_id, len(clip))
        if len(clip) == 0:
            continue
        with open(save_path + clip.item() + '.txt', 'w') as f:
            for line in lines:
                words = line.split()
                f.write(words[0] + ',' + words[1] + ',' + words[2] + ',' + words[3] + ',' + words[4] + ',' + words[5] +
                        ',' + words[6] + ',' + words[7] + ',' + words[8] + ',' + words[9] + ',' + ' '.join(words[10:]) + '\n')
        f.close()

def check_overlap(head_box, obj_curr, img_name):
    img = cv2.imread(img_name)
    max_left = max(head_box[0], obj_curr[0])
    max_top = max(head_box[1], obj_curr[1])
    min_right = min(head_box[2], obj_curr[2])
    min_bottom = min(head_box[3], obj_curr[3])

    if (min_right - max_left) > 0 and (min_bottom - max_top) > 0:
        # if (min_right - max_left)*(min_bottom - max_top)/((head_box[2] - head_box[0])*(head_box[3] - head_box[1])) > 0.8:
        return True
    # cv2.rectangle(img, (int(max_left), int(max_top)), (int(min_right), int(min_bottom)), (0, 0, 255),
    #               thickness=3)
    # cv2.rectangle(img, (int(head_box[0]), int(head_box[1])), (int(head_box[2]), int(head_box[3])), (255, 0, 0),
    #               thickness=3)
    # cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])), (int(obj_curr[2]), int(obj_curr[3])), (255, 0, 0),
    #               thickness=3)
    # cv2.imshow('img', img)
    # cv2.waitKey(2000)
    return False

def skeleton_person_check():
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    tracker_bbox_path = '../3d_pose2gaze/tracker_record_bbox/'
    img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    clips = os.listdir(annotation_path)
    person_dict = {}
    # clips = ['test_9434_3.txt']
    for clip in clips:
        img_names = sorted(glob.glob(img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        annt = pd.read_csv(annotation_path + clip, sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        with open(tracker_bbox_path + clip.split('.')[0] + '.p', 'rb') as f:
            tracker_bbox = joblib.load(f)
        first_frame = tracker_bbox.keys()[0]
        skeleton_bbox = tracker_bbox[first_frame][0][0]
        annt_box = annt.loc[(annt['frame'] == first_frame) & (annt['name'] == 'P1')]
        annt_box = [annt_box['x_min'].item(), annt_box['y_min'].item(), annt_box['x_max'].item(), annt_box['y_max'].item()]
        if not check_overlap(annt_box, skeleton_bbox, img_names[first_frame]):
            person_dict[clip.split('.')[0]] = 'P2'
        else:
            person_dict[clip.split('.')[0]] = 'P1'
    with open('person_id.p', 'wb') as f:
        pickle.dump(person_dict, f)

'''
input: [event, obj_location, memory_location, attention_mat]
output: [mind_change]
'''
def get_train_data():
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    mind_set_path = '/home/shuwen/data/data_preprocessing2/store_mind_set/'
    clips = os.listdir(event_label_path)
    for clip in clips:
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        with open(mind_set_path + clip, 'rb') as f:
            mind_sets = pickle.load(f)
        for event_seg in event_segs:
            frame_record, event_vec = event_seg

            # obj_location
            # find change frame


def check_append(obj_name, m1, mind_name, obj_frame, flags, label):
    if label:
        if not obj_name in m1:
            m1[obj_name] = []
            m1[obj_name].append([obj_frame.frame, [obj_frame.x_min, obj_frame.y_min, obj_frame.x_max, obj_frame.y_max], 0])
            flags[mind_name] = 1
        elif not flags[mind_name]:
            m1[obj_name].append([obj_frame.frame, [obj_frame.x_min, obj_frame.y_min, obj_frame.x_max, obj_frame.y_max], 0])
    else:
        if flags[mind_name]:
            m1[obj_name].append([obj_frame.frame, [obj_frame.x_min, obj_frame.y_min, obj_frame.x_max, obj_frame.y_max], 1])
            flags[mind_name] = 0
    return flags, m1

def store_mind_set(clip):
    #annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    annotation_path = '/home/lfan/Dropbox/Projects/ECCV20/reformat_annotation/'
    #save_path = '/home/shuwen/data/data_preprocessing2/store_mind_set/'
    save_path='/home/lfan/Dropbox/Projects/ECCV20/store_mind_set/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    annt = pd.read_csv(annotation_path + clip, sep=",", header=None)
    annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name", "label"]
    obj_names = annt.name.unique()
    m1, m2, m12, m21, mc = {}, {}, {}, {}, {}
    flags = {'m1':0, 'm2':0, 'm12':0, 'm21':0, 'mc':0}
    for obj_name in obj_names:
        if obj_name == 'P1' or obj_name == 'P2':
            continue
        obj_frames = annt.loc[annt.name == obj_name]
        for index, obj_frame in obj_frames.iterrows():
            if type(obj_frame.label) == float:
                # print obj_frame.label
                continue
            labels = obj_frame.label.split()
            for label in labels:
                if label == 'in_m1' or label == '"in_m1"':
                    flags, m1 = check_append(obj_name, m1, 'm1', obj_frame, flags, 1)
                elif label == 'in_m2' or label == '"in_m2"':
                    flags, m2 = check_append(obj_name, m2, 'm2', obj_frame, flags, 1)
                elif label == 'in_m12'or label == '"in_m12"':
                    flags, m12 = check_append(obj_name, m12, 'm12', obj_frame, flags, 1)
                elif label == 'in_m21' or label == '"in_m21"':
                    flags, m21 = check_append(obj_name, m21, 'm21', obj_frame, flags, 1)
                elif label == 'in_mc'or label == '"in_mc"':
                    flags, mc = check_append(obj_name, mc, 'mc', obj_frame, flags, 1)
                elif label == 'm1_false' or label == '"m1_false"':
                    flags, m1 = check_append(obj_name, m1, 'm1', obj_frame, flags, 0)
                elif label == 'm2_false' or label == '"m2_false"':
                    flags, m2 = check_append(obj_name, m2, 'm2', obj_frame, flags, 0)
                elif label == 'm12_false' or label == '"m12_false"':
                    flags, m12 = check_append(obj_name, m12, 'm12', obj_frame, flags, 0)
                elif label == 'm21_false' or label == '"m21_false"':
                    flags, m21 = check_append(obj_name, m2, 'm21', obj_frame, flags, 0)
    # print('m1', m1)
    # print('m2', m2)
    # print('m12', m12)
    # print('m21', m21)
    # print('mc', mc)
    with open(save_path + clip.split('.')[0] + '.p', 'wb') as f:
        pickle.dump([m1, m2, m12, m21, mc], f)




if __name__ == '__main__':
    #reframe_annotation()
    # skeleton_person_check()
    annotation_path = '/home/lfan/Dropbox/Projects/ECCV20/reformat_annotation/' #'/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    clips = os.listdir(annotation_path)
    for clip in clips:
        print(clip)
        store_mind_set(clip)






