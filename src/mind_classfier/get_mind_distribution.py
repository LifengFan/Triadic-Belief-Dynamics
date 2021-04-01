import os
import pickle
import numpy as np
import sys
sys.path.append('../data_processing_scripts/')
from metadata import *
from itertools import product

def seg2frame(segs):
    frames = np.empty((1, 0))
    for seg in segs:
        start = seg[0]
        end = seg[1]
        frame = np.ones((1, end - start + 1))*seg[2]
        frames = np.hstack([frames, frame])
    assert frames.shape[1] == segs[-1][1] + 1
    return frames

def get_data(data_path):
    event_count = {0: 0, 1: 0, 2: 0}
    event0_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event1_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event2_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event_hash = {0: event0_count, 1: event1_count, 2: event2_count}
    mind_combination = {0:{}, 1:{}, 2:{}}
    labels = []
    frame_ids = []
    clips_id = []
    clips=os.listdir(data_path)
    mind_dict = {'m1':0, 'm2':1, 'm12':2, 'm21':3, 'mc':4}
    for clip in clips:
        if clip not in clips_with_gt_event:
            continue
        print(clip)
        event_segs = event_seg_battery[clip.split('.')[0]]
        event_by_frames = seg2frame(event_segs)
        with open(data_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        obj_names = obj_records.keys()
        for obj_name in obj_names:
            for frame_id in range(len(obj_records[obj_name])):
                event = event_by_frames[0][frame_id]
                obj_record = obj_records[obj_name][frame_id]
                key = [0, 0, 0, 0, 0]
                for mind_name in obj_record.keys():
                    if mind_name == 'mg':
                        continue
                    mind_change = obj_record[mind_name]['fluent']
                    # if event == 0 and (mind_name == 'm12' or mind_name == 'm21' or mind_name == 'mc') and mind_change < 3:
                    #     print(clip, frame_id, obj_name, event, mind_name, mind_change)
                    event_hash[event][mind_name][mind_change] += 1
                    key[mind_dict[mind_name]] = mind_change

                key = tuple(key)
                if key in mind_combination[event]:
                    mind_combination[event][key] += 1
                else:
                    mind_combination[event][key] = 1

    print(event0_count)
    print(event1_count)
    print(event2_count)

    with open('./distribution_record/mind_sep.p', 'wb') as f:
        pickle.dump([event0_count, event1_count, event2_count], f)
    with open('./distribution_record/mind_combination_dis.p', 'wb') as f:
        pickle.dump(mind_combination, f)


    #         if model_type == 'single':
    #             vec_input, label_, frame_id = pickle.load(f)
    #             labels = labels + label_
    #             frame_ids = frame_ids + frame_id
    #             for i in range(len(label_)):
    #                 clips_id.append(clip)
    # assert len(labels) == len(frame_ids) == len(clips_id)
    # return labels, frame_ids, clips_id

def get_mind_transition(data_path):
    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    mind_matrix = list(product(mind_combination, repeat=2))
    event0_count = {key: 0 for key in mind_matrix}
    event1_count = {key: 0 for key in mind_matrix}
    event2_count = {key: 0 for key in mind_matrix}
    event0_mat = np.zeros((1024, 1024))
    event1_mat = np.zeros((1024, 1024))
    event2_mat = np.zeros((1024, 1024))
    event_hash = {0: event0_count, 1: event1_count, 2: event2_count}
    event_mat_hash = {0: event0_mat, 1: event1_mat, 2: event2_mat}

    mind_dict = {'m1': 0, 'm2': 1, 'm12': 2, 'm21': 3, 'mc': 4}
    for clip in clips_with_gt_event:

        print(clip)
        event_segs = event_seg_battery[clip.split('.')[0]]
        event_by_frames = seg2frame(event_segs)
        if not os.path.exists(data_path + clip):
            continue
        with open(data_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        obj_names = obj_records.keys()
        for obj_name in obj_names:
            last_key = None
            for frame_id in range(len(obj_records[obj_name])):
                event = event_by_frames[0][frame_id]
                obj_record = obj_records[obj_name][frame_id]
                key = [0, 0, 0, 0, 0]
                for mind_name in obj_record.keys():
                    if mind_name == 'mg':
                        continue
                    mind_change = obj_record[mind_name]['fluent']
                    key[mind_dict[mind_name]] = mind_change

                key = tuple(key)
                if frame_id == 0:
                    last_key = key
                else:
                    event_hash[event][(last_key, key)] += 1
                    event_mat_hash[event][mind_combination.index(last_key), mind_combination.index(key)] += 1

    count = sum(event0_count.values())
    for key in event0_count.keys():
        event0_count[key] = event0_count[key]/float(count)
    count = sum(event1_count.values())
    for key in event1_count.keys():
        event1_count[key] = event1_count[key] / float(count)
    count = sum(event2_count.values())
    for key in event2_count.keys():
        event2_count[key] = event2_count[key] / float(count)

    for i in range(1024):
        if np.sum(event0_mat[i]) > 0:
            event0_mat[i] = event0_mat[i]/np.sum(event0_mat[i])
        else:
            event0_mat[i] = 1./1024
        if np.sum(event1_mat[i]) > 0:
            event1_mat[i] = event1_mat[i]/np.sum(event1_mat[i])
        else:
            event1_mat[i] = 1./1024
        if np.sum(event2_mat[i]) > 0:
            event2_mat[i] = event2_mat[i]/np.sum(event2_mat[i])
        else:
            event2_mat[i] = 1./1024


    with open('./distribution_record/mind_transition_normed.p', 'wb') as f:
        pickle.dump([event0_count, event1_count, event2_count], f)

    with open('./distribution_record/mind_transition_mat.p', 'wb') as f:
        pickle.dump([event0_mat, event1_mat, event2_mat], f)

def get_mind_transition_split(data_path):

    event0_count = {key: 0 for key in range(4)}
    event1_count = {key: 0 for key in range(4)}
    event2_count = {key: 0 for key in range(4)}
    event0_mat = np.zeros((4, 4))
    event1_mat = np.zeros((4, 4))
    event2_mat = np.zeros((4, 4))
    event_hash = {0: event0_count, 1: event1_count, 2: event2_count}
    event_mat_hash = {0: event0_mat, 1: event1_mat, 2: event2_mat}

    mind_dict = {'m1': 0, 'm2': 1, 'm12': 2, 'm21': 3, 'mc': 4}
    for clip in clips_with_gt_event:

        print(clip)
        event_segs = event_seg_battery[clip.split('.')[0]]
        event_by_frames = seg2frame(event_segs)
        if not os.path.exists(data_path + clip):
            continue
        with open(data_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        obj_names = obj_records.keys()
        for obj_name in obj_names:
            last_key = None
            for frame_id in range(len(obj_records[obj_name])):
                event = event_by_frames[0][frame_id]
                obj_record = obj_records[obj_name][frame_id]
                key = [0, 0, 0, 0, 0]
                for mind_name in obj_record.keys():
                    if mind_name == 'mg':
                        continue
                    mind_change = obj_record[mind_name]['fluent']
                    key[mind_dict[mind_name]] = mind_change

                key = tuple(key)
                if frame_id == 0:
                    last_key = key
                else:
                    event_hash[event][(last_key, key)] += 1
                    event_mat_hash[event][mind_combination.index(last_key), mind_combination.index(key)] += 1

    count = sum(event0_count.values())
    for key in event0_count.keys():
        event0_count[key] = event0_count[key]/float(count)
    count = sum(event1_count.values())
    for key in event1_count.keys():
        event1_count[key] = event1_count[key] / float(count)
    count = sum(event2_count.values())
    for key in event2_count.keys():
        event2_count[key] = event2_count[key] / float(count)

    for i in range(1024):
        if np.sum(event0_mat[i]) > 0:
            event0_mat[i] = event0_mat[i]/np.sum(event0_mat[i])
        else:
            event0_mat[i] = 1./1024
        if np.sum(event1_mat[i]) > 0:
            event1_mat[i] = event1_mat[i]/np.sum(event1_mat[i])
        else:
            event1_mat[i] = 1./1024
        if np.sum(event2_mat[i]) > 0:
            event2_mat[i] = event2_mat[i]/np.sum(event2_mat[i])
        else:
            event2_mat[i] = 1./1024


    with open('./distribution_record/mind_transition_normed.p', 'wb') as f:
        pickle.dump([event0_count, event1_count, event2_count], f)

    with open('./distribution_record/mind_transition_mat.p', 'wb') as f:
        pickle.dump([event0_mat, event1_mat, event2_mat], f)





def get_event(event_seg, frame_id):
    for seg in event_seg:
        if frame_id >= seg[0] and frame_id <= seg[1]:
            return seg[2]


def calculate_distribution(data_y, data_clip, data_frame_id, split):
    event_count = {0:0, 1:0, 2:0}
    event0_count = {'m1':{0:0, 1:0, 2:0, 3:0}, 'm2':{0:0, 1:0, 2:0, 3:0}, 'm12':{0:0, 1:0, 2:0, 3:0},
                    'm21':{0:0, 1:0, 2:0, 3:0}, 'mc':{0:0, 1:0, 2:0, 3:0}}
    event1_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event2_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event_hash = {0: event0_count, 1: event1_count, 2: event2_count}

    for clip_id, clip in enumerate(data_clip):
        print(clip)
        clip_name = clip.split('.')[0]
        frame_id = data_frame_id[clip_id]
        mind_change = np.array(data_y[clip_id]).reshape(-1)
        if not clip_name in event_seg_tracker:
            continue
        tracker_event_seg = event_seg_tracker[clip_name]
        event = get_event(tracker_event_seg, frame_id)
        assert event is not None
        event_count[event] += 1

        print(mind_change)
        labels_mc = np.argmax(mind_change[:4])
        labels_m21 = np.argmax(mind_change[4:8])
        labels_m12 = np.argmax(mind_change[8:12])
        labels_m1 = np.argmax(mind_change[12:16])
        labels_m2 = np.argmax(mind_change[16:20])

        event_hash[event]['mc'][labels_mc] += 1
        event_hash[event]['m21'][labels_m21] += 1
        event_hash[event]['m12'][labels_m12] += 1
        event_hash[event]['m1'][labels_m1] += 1
        event_hash[event]['m2'][labels_m2] += 1

    with open('./distribution_record/' + split + '.p', 'wb') as f:
        pickle.dump([event_count, event0_count, event1_count, event2_count], f)

    print(event0_count, event1_count, event2_count)

def calculate_between_mind_distribution(data_y, split):
    mind_combination = {}
    for mind_change in data_y:
        mind_change = np.array(mind_change).reshape(-1)
        labels_mc = np.argmax(mind_change[:4])
        labels_m21 = np.argmax(mind_change[4:8])
        labels_m12 = np.argmax(mind_change[8:12])
        labels_m1 = np.argmax(mind_change[12:16])
        labels_m2 = np.argmax(mind_change[16:20])
        key = (labels_m1, labels_m2, labels_m12, labels_m21, labels_mc)
        if key in mind_combination:
            mind_combination[key] += 1
        else:
            mind_combination[key] = 1

    print(mind_combination)
    with open('./distribution_record/' + split + '.p', 'wb') as f:
        pickle.dump(mind_combination, f)

def normalize_event_transition():
    with open('./distribution_record/transition.p', 'rb') as f:
        transitions = pickle.load(f)

    print(transitions)
    counts =  sum(list(transitions.values()))
    for key in transitions.keys():
        transitions[key] = transitions[key]/float(counts)
    print(transitions)

    with open('./distribution_record/event_transition_normed.p', 'wb') as f:
        pickle.dump(transitions, f)



if __name__ == '__main__':
    data_path = './regenerate_annotation/'

    # mind distribution under different events
    calculate_distribution(labels, clips_id, frame_ids, 'event_mind_prob')
    # mind combination prior
    calculate_between_mind_distribution(labels, 'between_mind_prob')
    normalize_event_transition()
    # mind transition prior
    get_mind_transition(data_path)