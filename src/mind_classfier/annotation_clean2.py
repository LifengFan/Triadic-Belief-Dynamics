import pandas as pd
import os
import glob
import joblib
import cv2
import pickle
import numpy as np
import torch
import sys
sys.path.append('../attention_classifier')
from utils import *
from AttMat import AttMat

def reframe_annotation():
    annotation_path = '/home/shuwen/Downloads/all/'
    save_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
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

def get_grid_location(obj_frame):
    x_min = obj_frame['x_min']
    y_min = obj_frame['y_min']
    x_max = obj_frame['x_max']
    y_max = obj_frame['y_max']
    gridLW = 1280 / 25.
    gridLH = 720 / 15.
    center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
    X, Y = int(center_x / gridLW), int(center_y / gridLH)
    return X, Y

def regenerate_annotation():
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    save_path='/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tasks = glob.glob(annotation_path + '*.txt')
    for task in tasks:
        # if not task.split('/')[-1] == 'test_94342_20.txt':
        #     continue
        print(task)
        annt = pd.read_csv(task, sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name", "label"]
        obj_records = {}
        for index, obj_frame in annt.iterrows():
            if obj_frame['name'].startswith('P'):
                continue
            else:
                assert obj_frame['name'].startswith('O')
            obj_name = obj_frame['name']
            # 0: enter 1: disappear 2: update 3: unchange

            frame_id = obj_frame['frame']
            curr_loc = get_grid_location(obj_frame)
            mind_dict = {'m1': {'fluent': 3, 'loc': None}, 'm2': {'fluent': 3, 'loc': None},
                         'm12': {'fluent': 3, 'loc': None},
                         'm21': {'fluent': 3, 'loc': None}, 'mc': {'fluent': 3, 'loc': None},
                         'mg': {'fluent': 3, 'loc': curr_loc}}
            mind_dict['mg']['loc'] = curr_loc
            if not type(obj_frame['label']) == float:
                mind_labels = obj_frame['label'].split()
                for mind_label in mind_labels:
                    if mind_label == 'in_m1' or mind_label == 'in_m2' or mind_label == 'in_m12' \
                        or mind_label == 'in_m21' or mind_label == 'in_mc' or mind_label == '"in_m1"' or mind_label == '"in_m2"'\
                            or mind_label == '"in_m12"' or mind_label == '"in_m21"' or mind_label == '"in_mc"':
                        mind_name = mind_label.split('_')[1].split('"')[0]
                        mind_dict[mind_name]['loc'] = curr_loc
                    else:
                        mind_name = mind_label.split('_')[0].split('"')
                        if len(mind_name) > 1:
                            mind_name = mind_name[1]
                        else:
                            mind_name = mind_name[0]
                        last_loc = obj_records[obj_name][frame_id - 1][mind_name]['loc']
                        mind_dict[mind_name]['loc'] = last_loc

            for mind_name in mind_dict.keys():
                if frame_id > 0:
                    curr_loc = mind_dict[mind_name]['loc']
                    last_loc = obj_records[obj_name][frame_id - 1][mind_name]['loc']
                    if last_loc is None and curr_loc is not None:
                        mind_dict[mind_name]['fluent'] = 0
                    elif last_loc is not None and curr_loc is None:
                        mind_dict[mind_name]['fluent'] = 1
                    elif not last_loc ==  curr_loc:
                        mind_dict[mind_name]['fluent'] = 2
            if obj_name not in obj_records:
                obj_records[obj_name] = [mind_dict]
            else:
                obj_records[obj_name].append(mind_dict)

        # for frame_id in range(len(obj_records['O1'])):
        #     print(frame_id, obj_records['O1'][frame_id])
        with open(save_path + task.split('/')[-1].split('.')[0] + '.p', 'wb') as f:
            pickle.dump(obj_records, f)

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

def find_other_mind(mind_sets, obj_name, change_frame_id):
    frames = {}
    for mind_id, mind_set in enumerate(mind_sets):
        if obj_name in mind_set:
            record_frames = mind_set[obj_name]
            for record_frame in record_frames:
                if record_frame[0]>=change_frame_id - 2 and record_frame[0] <= change_frame_id + 2:
                    frame_id = record_frame[0] - change_frame_id + 2
                    if frame_id in frames:
                        frames[frame_id].append([mind_id, record_frame[2]])
                    else:
                        frames[frame_id] = [[mind_id, record_frame[2]]]
    return frames

def find_memory_bbox(mind_sets, change_frame_id, obj_name):
    memory_bboxs = []
    for mind_set in mind_sets:
        memory_bbox = None
        if obj_name in mind_set:
            obj_frames = mind_set[obj_name]
            for obj_frame in obj_frames:
                if obj_frame[0] < change_frame_id:
                    memory_bbox = obj_frame[1]
                else:
                    break
        memory_bboxs.append(memory_bbox)
    return memory_bboxs

def get_attention_input(p1_patch, p1_bbox, p2_bbox, curr_bbox, memory_bboxs, img):
    p1_head_patch = []
    width = float(img.shape[1])
    height = float(img.shape[0])
    p1_patch = cv2.resize(p1_patch, (224, 224)).reshape((3, 224, 224))
    for c in [0, 1, 2]:
        p1_patch[c, :, :] = (p1_patch[c, :, :] / 255. - 0.5) / 0.5
    head_pos = np.array([p1_bbox[0] / width, p1_bbox[1] /height, p1_bbox[2] / width,
                         p1_bbox[3] / height, (p1_bbox[0] + p1_bbox[2]) / 2 / width,
                         (p1_bbox[1] + p1_bbox[3]) / 2 / height])
    cv2.rectangle(img, (p1_bbox[0], p1_bbox[1]), (p1_bbox[2], p1_bbox[3]), (255, 0, 0), thickness=2)
    p1_pos_vec = np.empty((0, 12))

    # add p2
    vec = np.array([p2_bbox[0] / width, p2_bbox[1] / height, p2_bbox[2] / width,
                    p2_bbox[3] / height, (p2_bbox[0] + p2_bbox[2]) / 2 /width,
                    (p2_bbox[1] + p2_bbox[3]) / 2 / height])
    cv2.rectangle(img, (p2_bbox[0], p2_bbox[1]), (p2_bbox[2], p2_bbox[3]), (255, 0, 0), thickness=2)
    vec = np.hstack(([head_pos, vec]))
    p1_head_patch.append(p1_patch)
    p1_pos_vec = np.vstack([p1_pos_vec, vec])

    # add currbbox
    p1_head_patch.append(p1_patch)
    vec = np.array([curr_bbox[0] / width, curr_bbox[1] /height, curr_bbox[2] / width,
                    curr_bbox[3] / height, (curr_bbox[0] + curr_bbox[2]) / 2 / width,
                    (curr_bbox[1] + curr_bbox[3]) / 2 /height])
    cv2.rectangle(img, (curr_bbox[0], curr_bbox[1]), (curr_bbox[2], curr_bbox[3]), (255, 0, 0), thickness=2)
    vec = np.hstack([head_pos, vec])
    p1_pos_vec = np.vstack([p1_pos_vec, vec])

    # add memory bbox
    memory_mark = {}
    for mind_id, memory_bbox in enumerate(memory_bboxs):
        if memory_bbox:
            vec = np.array([memory_bbox[0] / width, memory_bbox[1] / height, memory_bbox[2] / width,
                            memory_bbox[3] / height, (memory_bbox[0] + memory_bbox[2]) / 2 / width,
                            (memory_bbox[1] + memory_bbox[3]) / 2 / height])
            memory_mark[mind_id] = 1
            cv2.rectangle(img, (memory_bbox[0], memory_bbox[1]), (memory_bbox[2], memory_bbox[3]), (0, 0, 255), thickness=2)
        else:
            vec = np.zeros(6)
        vec = np.hstack(([head_pos, vec]))
        p1_head_patch.append(p1_patch)
        p1_pos_vec = np.vstack([p1_pos_vec, vec])
    # cv2.imshow('img', img)
    # cv2.waitKey(2000)
    return p1_head_patch, p1_pos_vec, memory_mark

def get_attention_output(p1_head_patches, p1_pos_vecs, net, attmat, pid, memory_mark):
    heads = np.array(p1_head_patches).astype(float)
    for c in [0, 1, 2]:
        heads[:, c, :, :] = (heads[:, c, :, :] / 255. - 0.5) / 0.5
    heads = torch.from_numpy(heads).float().cuda()
    # print(p1_pos_vecs)
    pos = torch.from_numpy(p1_pos_vecs).float().cuda()
    predicted_val = net(torch.autograd.Variable(heads), torch.autograd.Variable(pos))
    max_score, idx = torch.max(predicted_val, 1)
    idxs = idx.cpu().numpy()
    inds = np.where(idxs == 1)[0]
    if inds.shape[0] > 0:
        for ind in inds:
            if ind == 0:
                if pid == 0:
                    attmat[pid, 1] = 1
                else:
                    attmat[pid, 0] = 1
            else:
                if ind > 1:
                    if ind - 2 in memory_mark:
                        attmat[pid, ind + 1] = 1
                else:
                    attmat[pid, ind + 1] = 1

    return attmat

def find_training_input(mind_sets, battery_event_labels, tracker_event_labels, annt, color_img_names, net):
    frame_length = 5
    total_training_input = []
    total_vec_input = []
    total_training_output = []
    frame_record = []
    for mind_set in mind_sets:
        for obj_name in mind_set.keys():
            change_records = mind_set[obj_name]
            for change_record in change_records:
                change_frame_id, bbox, enter = change_record
                frames = find_other_mind(mind_sets, obj_name, change_frame_id)
                output = np.zeros(15)
                unchange_id = [2, 5, 8, 11, 14]
                output[unchange_id] = 1
                events = np.zeros((2, 25))
                locations = np.zeros((6, 10))
                attmats = np.zeros((5, 2, 8))
                indicator = np.zeros((5, 5))
                for i in range(frame_length):
                    if i < 2:
                        curr_frame_id = max(0, change_frame_id - 2 + i)
                    else:
                        curr_frame_id = min(battery_event_labels.shape[0], change_frame_id - 2 + i)
                    # output
                    if i in frames:
                        for mind_id, label in frames[i]:
                            if label == 0:
                                output[mind_id*3] = 1
                                output[mind_id + 2] = 0
                            else:
                                output[mind_id*3 + 1] = 1
                                output[mind_id*3 + 2] = 0


                    # events
                    events[0, i*5:i*5+5] = tracker_event_labels[curr_frame_id]
                    events[1, i*5:i*5+5] = battery_event_labels[curr_frame_id]

                    # locations
                    curr_annt = annt.loc[(annt.name == obj_name) & (annt.frame == curr_frame_id)]
                    print(curr_annt)
                    curr_bbox = [curr_annt.x_min.item(), curr_annt.y_min.item(), curr_annt.x_max.item(), curr_annt.y_max.item()]
                    curr_center = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
                    locations[0, i*2:i*2+2] = np.array(curr_center)
                    memory_bboxs = find_memory_bbox(mind_sets, curr_frame_id, obj_name)

                    for mind_id, memory_bbox in enumerate(memory_bboxs):
                        if memory_bbox:
                            memory_center = [(memory_bbox[0] + memory_bbox[2])/2, (memory_bbox[1] + memory_bbox[3])/2]
                            relative_direct = np.array(curr_center) - np.array(memory_center)
                            indicator[i, mind_id] = 1
                        else:
                            relative_direct = np.zeros(2)

                        locations[mind_id + 1, i*2:i*2+2] = relative_direct

                    # attmat
                    # [p1 head, obj bbx]
                    p1_annt = annt.loc[(annt.name == 'P1') & (annt.frame == curr_frame_id)]
                    p1_bbox = [p1_annt.x_min.item(), p1_annt.y_min.item(), p1_annt.x_max.item(),
                                 p1_annt.y_max.item()]
                    p2_annt = annt.loc[(annt.name == 'P2') & (annt.frame == curr_frame_id)]
                    p2_bbox = [p2_annt.x_min.item(), p2_annt.y_min.item(), p2_annt.x_max.item(),
                               p2_annt.y_max.item()]
                    img = cv2.imread(color_img_names[curr_frame_id])
                    p1_patch = img[p1_bbox[1]:p1_bbox[3], p1_bbox[0]:p1_bbox[2]]
                    p1_head_patches, p1_pos_vecs, p1_memory_mark = get_attention_input(p1_patch, p1_bbox,
                                                                       p2_bbox, curr_bbox, memory_bboxs, img)

                    # [p2 head, obj bbx]
                    p2_patch = img[p1_bbox[1]:p1_bbox[3], p1_bbox[0]:p1_bbox[2]]
                    p2_head_patches, p2_pos_vecs, p2_memory_mark = get_attention_input(p2_patch, p2_bbox, p1_bbox, curr_bbox, memory_bboxs, img)

                    attmat = np.zeros((2, 8))

                    # get attention output
                    attmat = get_attention_output(p1_head_patches, p1_pos_vecs, net, attmat, 0, p1_memory_mark)
                    attmat = get_attention_output(p2_head_patches, p2_pos_vecs, net, attmat, 1, p2_memory_mark)
                    # print(attmat)
                    attmats[i, :, :] = attmat

                total_training_input.append([events, locations, attmats, indicator])
                input_vec = np.hstack([events.reshape((1, -1)), locations.reshape((1, -1)), attmats.reshape((1, -1)),
                                       indicator.reshape((1, -1))])
                total_vec_input.append(input_vec)
                total_training_output.append(output)
                frame_record.append([change_frame_id])
    return total_training_input, total_vec_input, total_training_output, frame_record



def get_labels_by_frames(event_labels, labels_by_frames, video_length):
    for event_label in event_labels:
        for i in range(event_label[0][1] - event_label[0][0] + 1):
            if not event_label[1] == 'NA':
                labels_by_frames = np.vstack([labels_by_frames, event_label[1]])
            else:
                labels_by_frames = np.vstack([labels_by_frames, np.zeros(5)])
    if labels_by_frames.shape[0] <  video_length:
        for i in range(labels_by_frames.shape[0], video_length):
            labels_by_frames = np.vstack([labels_by_frames, np.zeros(5)])
    return labels_by_frames

def reformat_events(event_segs, video_length):
    # tracker
    tracker_labels_by_frames = np.empty((0, 5))
    event_labels = event_segs['tracker']
    tracker_labels_by_frames = get_labels_by_frames(event_labels, tracker_labels_by_frames, video_length)

    battery_labels_by_frames = np.empty((0, 5))
    event_labels = event_segs['battery']
    battery_labels_by_frames = get_labels_by_frames(event_labels, battery_labels_by_frames, video_length)

    assert tracker_labels_by_frames.shape == battery_labels_by_frames.shape
    return battery_labels_by_frames, tracker_labels_by_frames

'''
input: [event, obj_location, memory_location, attention_mat]
output: [mind_change]
'''
def get_train_data():
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    mind_set_path = '/home/shuwen/data/data_preprocessing2/store_mind_set/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    save_training_path = '/home/shuwen/data/data_preprocessing2/mind_training/'
    if not os.path.exists(save_training_path):
        os.makedirs(save_training_path)
    clips = os.listdir(event_label_path)
    net = AttMat()
    net = load_best_checkpoint(net, '/home/shuwen/data/Six-Minds-Project/attention_classifier/')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    for clip in clips:
        if not os.path.exists(mind_set_path + clip):
            continue
        # if os.path.exists(save_training_path + clip.split('.')[0] + '.p'):
        #     continue
        print(clip)
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs)
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        with open(mind_set_path + clip, 'rb') as f:
            mind_sets = pickle.load(f)
        color_img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        total_training_input, total_vec_input, total_training_output, frame_record = \
            find_training_input(mind_sets, p2_events_by_frame, p1_events_by_frame, annt, color_img_names, net)
        with open(save_training_path + clip.split('.')[0] + '.p', 'wb') as f:
            pickle.dump([total_training_input, total_vec_input, total_training_output, frame_record], f)


def check_append(obj_name, m1, mind_name, obj_frame, flags, label):
    if label:
        if not obj_name in m1:
            m1[obj_name] = []
            m1[obj_name].append(
                [obj_frame.frame, [obj_frame.x_min, obj_frame.y_min, obj_frame.x_max, obj_frame.y_max], 0])
            flags[mind_name] = 1
        elif not flags[mind_name]:
            m1[obj_name].append(
                [obj_frame.frame, [obj_frame.x_min, obj_frame.y_min, obj_frame.x_max, obj_frame.y_max], 0])
            flags[mind_name] = 1
    else:
        if obj_name in m1:
            if flags[mind_name]:
                m1[obj_name].append(
                    [obj_frame.frame, [obj_frame.x_min, obj_frame.y_min, obj_frame.x_max, obj_frame.y_max], 1])
                flags[mind_name] = 0
    return flags, m1

def store_mind_set(clip):
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    save_path = '/home/shuwen/data/data_preprocessing2/store_mind_set/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    annt = pd.read_csv(annotation_path + clip, sep=",", header=None)
    annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                    "label"]
    obj_names = annt.name.unique()
    m1, m2, m12, m21, mc = {}, {}, {}, {}, {}
    flags = {'m1':0, 'm2':0, 'm12':0, 'm21':0, 'mc':0}
    for obj_name in obj_names:
        if obj_name == 'P1' or obj_name == 'P2':
            continue
        obj_frames = annt.loc[annt.name == obj_name]
        for index, obj_frame in obj_frames.iterrows():
            if type(obj_frame.label) == float:
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
                    flags, m12 = check_append(obj_name, m12, 'm12', obj_frame, flags, 0)
                    flags, m21 = check_append(obj_name, m21, 'm21', obj_frame, flags, 0)
                elif label == 'm2_false' or label == '"m2_false"':
                    flags, m2 = check_append(obj_name, m2, 'm2', obj_frame, flags, 0)
                    flags, m12 = check_append(obj_name, m12, 'm12', obj_frame, flags, 0)
                    flags, m21 = check_append(obj_name, m21, 'm21', obj_frame, flags, 0)
                elif label == 'm12_false' or label == '"m12_false"':
                    flags, m12 = check_append(obj_name, m12, 'm12', obj_frame, flags, 0)
                    flags, mc = check_append(obj_name, mc, 'mc', obj_frame, flags, 0)
                elif label == 'm21_false' or label == '"m21_false"':
                    flags, m21 = check_append(obj_name, m2, 'm21', obj_frame, flags, 0)
                    flags, mc = check_append(obj_name, mc, 'mc', obj_frame, flags, 0)
    # print('m1', m1)
    # print('m2', m2)
    # print('m12', m12)
    # print('m21', m21)
    # print('mc', mc)
    with open(save_path + clip.split('.')[0] + '.p', 'wb') as f:
        pickle.dump([m1, m2, m12, m21, mc], f)

def find_objects(mind_sets, frame_id):
    objects = {}
    for mind_id, mind_set in enumerate(mind_sets):
        for obj_name in mind_set.keys():
            frame_records = mind_set[obj_name]
            for frame_record in frame_records:
                if frame_record[0] >= frame_id and frame_record[0] < frame_id + 5:
                    if obj_name in objects:
                        objects[obj_name].append([mind_id, frame_record])
                    else:
                        objects[obj_name] = [[mind_id, frame_record]]
    return objects

def get_test_data_input(p1_events_by_frame, p2_events_by_frame, mind_sets, annt, img_names, net):
    total_training_input = []
    total_vec_input = []
    total_training_output = []
    frame_record = []
    for frame_id in range(0, p1_events_by_frame.shape[0] - 5 + 1, 2):
        objects = find_objects(mind_sets, frame_id)
        if len(objects) == 0:
            continue
        for obj_name in objects.keys():
            output = np.zeros(15)
            unchange_id = [2, 5, 8, 11, 14]
            output[unchange_id] = 1
            events = np.zeros((2, 25))
            locations = np.zeros((6, 10))
            attmats = np.zeros((5, 2, 8))
            indicator = np.zeros((5, 5))

            # output
            for record in objects[obj_name]:
                mind_id, frame_record = record
                if frame_record[2] == 0:
                    output[mind_id * 3] = 1
                    output[mind_id + 2] = 0
                else:
                    output[mind_id * 3 + 1] = 1
                    output[mind_id * 3 + 2] = 0

            for i in range(5):
                curr_frame_id = frame_id + i
                # event
                events[0, i*5:i*5+5] = p1_events_by_frame[curr_frame_id]
                events[1, i*5:i*5+5] = p2_events_by_frame[curr_frame_id]

                # location
                curr_annt = annt.loc[(annt.name == obj_name) & (annt.frame == curr_frame_id)]
                curr_bbox = [curr_annt.x_min.item(), curr_annt.y_min.item(), curr_annt.x_max.item(),
                             curr_annt.y_max.item()]
                curr_center = [(curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2]
                locations[0, i * 2:i * 2 + 2] = np.array(curr_center)
                memory_bboxs = find_memory_bbox(mind_sets, curr_frame_id, obj_name)

                for mind_id, memory_bbox in enumerate(memory_bboxs):
                    if memory_bbox:
                        memory_center = [(memory_bbox[0] + memory_bbox[2]) / 2, (memory_bbox[1] + memory_bbox[3]) / 2]
                        relative_direct = np.array(curr_center) - np.array(memory_center)
                        indicator[i, mind_id] = 1
                    else:
                        relative_direct = np.zeros(2)

                    locations[mind_id + 1, i * 2:i * 2 + 2] = relative_direct

                # attmat
                # [p1 head, obj bbx]
                p1_annt = annt.loc[(annt.name == 'P1') & (annt.frame == curr_frame_id)]
                p1_bbox = [p1_annt.x_min.item(), p1_annt.y_min.item(), p1_annt.x_max.item(),
                           p1_annt.y_max.item()]
                p2_annt = annt.loc[(annt.name == 'P2') & (annt.frame == curr_frame_id)]
                p2_bbox = [p2_annt.x_min.item(), p2_annt.y_min.item(), p2_annt.x_max.item(),
                           p2_annt.y_max.item()]
                img = cv2.imread(img_names[curr_frame_id])
                p1_patch = img[p1_bbox[1]:p1_bbox[3], p1_bbox[0]:p1_bbox[2]]
                p1_head_patches, p1_pos_vecs, p1_memory_mark = get_attention_input(p1_patch, p1_bbox,
                                                                                   p2_bbox, curr_bbox, memory_bboxs,
                                                                                   img)

                # [p2 head, obj bbx]
                p2_patch = img[p1_bbox[1]:p1_bbox[3], p1_bbox[0]:p1_bbox[2]]
                p2_head_patches, p2_pos_vecs, p2_memory_mark = get_attention_input(p2_patch, p2_bbox, p1_bbox,
                                                                                   curr_bbox, memory_bboxs, img)

                attmat = np.zeros((2, 8))

                # get attention output
                attmat = get_attention_output(p1_head_patches, p1_pos_vecs, net, attmat, 0, p1_memory_mark)
                attmat = get_attention_output(p2_head_patches, p2_pos_vecs, net, attmat, 1, p2_memory_mark)
                # print(attmat)
                attmats[i, :, :] = attmat

            total_training_input.append([events, locations, attmats, indicator])
            input_vec = np.hstack([events.reshape((1, -1)), locations.reshape((1, -1)), attmats.reshape((1, -1)),
                                   indicator.reshape((1, -1))])
            total_vec_input.append(input_vec)
            total_training_output.append(output)
            frame_record.append([frame_id])
    return total_training_input, total_vec_input, total_training_output, frame_record





def get_test_data():
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    mind_set_path = '/home/shuwen/data/data_preprocessing2/store_mind_set/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_testing/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(event_label_path)
    net = AttMat()
    net = load_best_checkpoint(net, '/home/shuwen/data/Six-Minds-Project/attention_classifier/')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for clip in clips:
        if not os.path.exists(mind_set_path + clip):
            continue
        print(clip)
        with open(mind_set_path + clip, 'rb') as f:
            mind_sets = pickle.load(f)

        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs)
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        total_training_input, total_vec_input, total_training_output, frame_record = \
            get_test_data_input(p1_events_by_frame, p2_events_by_frame, mind_sets, annt, img_names, net)
        with open(save_path + clip.split('.')[0] + '.p', 'wb') as f:
            pickle.dump([total_training_input, total_vec_input, total_training_output, frame_record], f)

def get_retrain_data():
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_retraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(event_label_path)
    for clip in clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        training_input = []
        training_output = []
        for obj_name in obj_records.keys():
            obj_record = obj_records[obj_name]
            for frame_id, frame_record in enumerate(obj_record):
                for mind_name in frame_record.keys():
                    if not frame_record[mind_name]['fluent'] == 3:
                        # event
                        p1_event = p1_events_by_frame[frame_id]
                        p2_event = p2_events_by_frame[frame_id]
                        # curr_grid
                        output = np.zeros((6, 4))
                        curr_loc = get_grid_location(annt.loc[frame_id])
                        memory_dist = []
                        indicator = []
                        for mind_id, mind_name in enumerate(frame_record.keys()):
                            output[mind_id, frame_record[mind_name]['fluent']] = 1
                            if frame_id == 0:
                                memory_dist.append(0)
                                indicator.append(0)
                            else:
                                memory_loc = obj_record[frame_id - 1][mind_name]['loc']
                                if memory_loc:
                                    curr_loc = np.array(curr_loc)
                                    memory_loc = np.array(memory_loc)
                                    memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                                    indicator.append(1)
                                else:
                                    memory_dist.append(0)
                                    indicator.append(0)
                        memory_dist = np.array(memory_dist)
                        indicator = np.array(indicator)
                        input = np.hstack([p1_event.reshape((1, -1)), p2_event.reshape((1, -1)), memory_dist.reshape((1, -1)),
                                           indicator.reshape((1, -1))])
                        output = output.reshape((1, -1))
                        god_output = output[0, 3]
                        output = output[0, 4:]
                        output = np.append(output, god_output)

                        assert output.shape[0] == 21
                        training_input.append(input)
                        training_output.append(output)
                        # if frame_id == 501 and obj_name == 'O1':
                        #     print(input)
                        #     print(output)
                        break
        with open(save_path + clip, 'wb') as f:
            pickle.dump([training_input, training_output], f)

def interpolate_obj_bbox():
    bbox_path = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
    save_path = '/home/shuwen/data/data_preprocessing2/interpolate_bbox/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    clips = os.listdir(bbox_path)
    for clip in clips:
        print(clip)
        clip_save_path = save_path + clip + '/'
        if not os.path.exists(clip_save_path):
            os.makedirs(clip_save_path)
        obj_names = sorted(glob.glob(bbox_path + clip + '/*.p'))
        for obj_name in obj_names:
            print(obj_name)
            with open(obj_name, 'rb') as f:
                obj_frames = joblib.load(f)
            obj_new_record = np.empty((0, 4))
            temp_id = None
            for frame_id, obj_frame in enumerate(obj_frames):
                obj_frame = np.array(obj_frame)[:4]
                obj_new_record = np.vstack([obj_new_record, obj_frame])
                if np.mean(obj_frame == 0):
                    if temp_id is None:
                        temp_id = frame_id
                else:
                    if temp_id is not None:
                        if temp_id == 0:
                            for i in range(0, frame_id):
                                obj_new_record[i] = obj_new_record[frame_id]
                        else:
                            for i in range(temp_id, frame_id):
                                obj_new_record[i] = (obj_new_record[frame_id] - obj_new_record[temp_id - 1])/(frame_id - temp_id)*(i - temp_id + 1) \
                                                + obj_new_record[temp_id - 1]
                        temp_id = None
            if temp_id is not None:
                for i in range(temp_id, obj_new_record.shape[0]):
                    obj_new_record[i] = obj_new_record[temp_id - 1]
            with open(clip_save_path + obj_name.split('/')[-1], 'wb') as f:
                pickle.dump(obj_new_record, f)

def get_seq_data():
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_lstm_training/'
    seq_len = 5
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(event_label_path)
    for clip in clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        training_input = []
        training_output = []
        for obj_name in obj_records.keys():
            obj_record = obj_records[obj_name]
            for frame_id, frame_record in enumerate(obj_record):
                for mind_name in frame_record.keys():
                    if not frame_record[mind_name]['fluent'] == 3:
                        event_input = np.zeros((seq_len, 10))
                        memory_input = np.zeros((seq_len, 10))
                        output = np.zeros((5, 4))
                        for mind_id, mind_name in enumerate(frame_record.keys()):
                            if mind_name == 'mg':
                                continue
                            output[mind_id - 1, frame_record[mind_name]['fluent']] = 1
                        for i in range(-4, 1, 1):
                            curr_frame_id = max(frame_id + i, 0)
                            # event
                            p1_event = p1_events_by_frame[curr_frame_id]
                            p2_event = p2_events_by_frame[curr_frame_id]
                            event_input[i + 4, :5] = p1_event
                            event_input[i + 4, 5:] = p2_event
                            # curr_grid
                            curr_loc = get_grid_location(annt.loc[curr_frame_id])
                            memory_dist = []
                            indicator = []
                            for mind_id, mind_name in enumerate(frame_record.keys()):
                                if mind_name == 'mg':
                                    continue
                                if curr_frame_id == 0:
                                    memory_dist.append(0)
                                    indicator.append(0)
                                else:
                                    memory_loc = obj_record[curr_frame_id - 1][mind_name]['loc']
                                    if memory_loc:
                                        curr_loc = np.array(curr_loc)
                                        memory_loc = np.array(memory_loc)
                                        memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                                        indicator.append(1)
                                    else:
                                        memory_dist.append(0)
                                        indicator.append(0)
                            memory_dist = np.array(memory_dist)
                            indicator = np.array(indicator)
                            memory_input[i + 4, :5] = memory_dist
                            memory_input[i + 4, 5:] = indicator
                        input = np.hstack([event_input, memory_input])
                        output = output.reshape((1, -1))
                        assert output.shape[1] == 20
                        training_input.append(input)
                        training_output.append(output)
                        break
        with open(save_path + clip, 'wb') as f:
            pickle.dump([training_input, training_output], f)

if __name__ == '__main__':
    # reframe_annotation()
    regenerate_annotation()
    # skeleton_person_check()

    # annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    # clips = os.listdir(annotation_path)
    # for clip in clips:
    #     print(clip)
    #     store_mind_set(clip)

    # get_train_data()
    #
    # get_test_data()

    # get_retrain_data()

    # interpolate_obj_bbox()

    get_seq_data()