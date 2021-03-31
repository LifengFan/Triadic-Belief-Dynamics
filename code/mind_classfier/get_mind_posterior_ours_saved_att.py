import sys
sys.path.append('../data_processing_scripts/')
from annotation_clean import *
from mind_model_att import *
from metadata import *
import os.path as op
import pickle
import sklearn.metrics as metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import product
import torch
import glob
import copy
from sklearn.preprocessing import OneHotEncoder

def seg2frame_score(tracker_seg):
    event_frame = np.zeros((tracker_seg['cp'][-1], 3))
    for idx in range(len(tracker_seg['cp']) - 1):
        start = tracker_seg['cp'][idx]
        end = tracker_seg['cp'][idx + 1]
        # event = np.zeros((1, 3))
        # event[0][tracker_seg['event'][idx][0]] = 1
        event=tracker_seg['event_vec'][idx]

        for i in range(start, end):
            event_frame[i, :] = event
    return event_frame

def check_overlap_return_area(head_box, obj_curr):
    max_left = max(head_box[0], obj_curr[0])
    max_top = max(head_box[1], obj_curr[1])
    min_right = min(head_box[2], obj_curr[2])
    min_bottom = min(head_box[3], obj_curr[3])
    if (min_right - max_left) > 0 and (min_bottom - max_top) > 0:
        return (min_right - max_left)*(min_bottom - max_top)
    return -100

def get_grid_location_using_bbox(obj_frame):
    x_min = obj_frame[0]
    y_min = obj_frame[1]
    x_max = obj_frame[0] + obj_frame[2]
    y_max = obj_frame[1] + obj_frame[3]
    gridLW = 1280 / 25.
    gridLH = 720 / 15.
    center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
    X, Y = int(center_x / gridLW), int(center_y / gridLH)
    return X, Y

def get_obj_name(obj_bbox, annt, frame_id):
    obj_candidates = annt.loc[annt.frame == frame_id]
    max_overlap = 0
    max_name = None
    max_bbox = None
    obj_bbox = [obj_bbox[0], obj_bbox[1], obj_bbox[0] + obj_bbox[2], obj_bbox[1] + obj_bbox[3]]
    obj_area = (obj_bbox[2] - obj_bbox[0])*(obj_bbox[3] - obj_bbox[1])
    for index, obj_candidate in obj_candidates.iterrows():
        if obj_candidate['name'].startswith('P'):
            continue
        if obj_candidate['lost'] == 1:
            continue
        candidate_bbox = [obj_candidate['x_min'], obj_candidate['y_min'], obj_candidate['x_max'], obj_candidate['y_max']]
        candidate_area = (obj_candidate['x_max'] - obj_candidate['x_min'])*(obj_candidate['y_max'] - obj_candidate['y_min'])
        overlap = check_overlap_return_area(obj_bbox, candidate_bbox)
        if overlap > max_overlap and overlap/obj_area < 1.2 and overlap/obj_area > 0.3 and overlap/candidate_area <1.2 and overlap/candidate_area>0.3:
            max_overlap = overlap
            max_name = obj_candidate['name']
            max_bbox = candidate_bbox
    if max_overlap > 0:
        return max_name, max_bbox
    return None, None

def update_memory(memory, mind_name, fluent, loc):

    if fluent == 0 or fluent == 2:
        memory[mind_name]['loc'] = loc
    elif fluent == 1:
        memory[mind_name]['loc'] = None

    return memory

def get_posterior(likelihood, mind_prior, transition_prior, eid1, eid2):
    assert abs(np.sum(likelihood) - 1) < 0.0001
    assert abs(np.sum(mind_prior[eid1]) - 1) < 0.0001
    assert abs(np.sum(mind_prior[eid2]) - 1) < 0.0001
    assert abs(np.sum(mind_prior[eid1].dot(transition_prior[eid1]) - 1) < 0.0001)

    pos1 = likelihood * (mind_prior[eid1].dot(transition_prior[eid1]))
    pos2 = likelihood * (mind_prior[eid2].dot(transition_prior[eid2]))
    pos = (pos2 + pos1)/2
    pos = pos/np.sum(pos)
    return pos

def get_gt_data():
    reannotation_path = './regenerate_annotation/'
    annotation_path = './reformat_annotation/'
    color_img_path = './annotations/'
    hog_path = './feature_single/'
    att_path = './att_vec/'
    model_path = './cptk/model_best.pth'
    save_path = './mind_full_likelihood_new_att/'
    tree_path =  './BestTree_full_0915/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(op.join('./', 'distribution_record/mind_combination_dis.p'), 'rb') as f:
        event_mind_prob = pickle.load(f)

    for event in event_mind_prob.keys():
        event_prob = event_mind_prob[event]
        mind_count = np.zeros(1024)
        mind_combination = list(product([0, 1, 2, 3], repeat=5))
        for cid, combination in enumerate(mind_combination):
            if combination in event_prob:
                mind_count[cid] = event_prob[combination]
        mind_count = np.array(mind_count)
        mind_count = mind_count / float(np.sum(mind_count))
        assert abs(np.sum(mind_count) - 1) < 0.001
        event_mind_prob[event] = mind_count

    with open(op.join('./', 'distribution_record/mind_transition_mat.p'), 'rb') as f:
        mind_transition_prob = pickle.load(f)

    clips = mind_test_clips
    # clips = clips_with_gt_event
    with open(op.join('./', 'person_id.p'), 'rb') as f:
        person_ids = pickle.load(f)

    net = MLP_Combined_Att()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    for clip in clips:
        clip_name = clip.split('.')[0]
        if not os.path.exists(reannotation_path + clip):
            continue
        with open(op.join(tree_path, clip), 'rb') as f:
            tree = pickle.load(f)

        print(clip)
        save_prefix = save_path + clip.split('.')[0] + '/'
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)


        tracker_seg = tree['T']
        battery_seg = tree['B']

        tracker_events_by_frame = seg2frame_score(tracker_seg)
        battery_events_by_frame = seg2frame_score(battery_seg)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        # tracker_events_by_frame = reformat_events_gt(event_seg_tracker[clip_name],
        #                                              len(img_names))
        # battery_events_by_frame = reformat_events_gt(event_seg_battery[clip_name],
        #                                              len(img_names))
        assert  tracker_events_by_frame.shape == battery_events_by_frame.shape

        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(hog_path + clip, 'rb') as f:
            features = pickle.load(f)
        with open(att_path + clip, 'rb') as f:
            att_vec = pickle.load(f)


        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
            p1_hog = features[1]
            p2_hog = features[2]

        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
            p1_hog = features[2]
            p2_hog = features[1]


        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name","label"]
        obj_names = annt.name.unique()

        mind_dict = {'m1': 0, 'm2': 1, 'm12': 2, 'm21': 3, 'mc': 4}
        for obj_name in obj_names:
            # if os.path.exists(save_prefix + obj_name.split('/')[-1] + '.p'):
            #     continue
            if obj_name.startswith('P'):
                continue
            print(obj_name)

            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            mind_output_gt = []
            mind_output = []
            for frame_id in range(p1_events_by_frame.shape[0]):
                # curr_loc
                curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                curr_loc = get_grid_location(curr_df)

                obj_record = obj_records[obj_name][frame_id]
                mind_output_gt.append(obj_record)

                # event

                output = att_vec[obj_name][frame_id]
                p1_event = p1_events_by_frame[frame_id]
                p2_event = p2_events_by_frame[frame_id]

                # memory
                memory_dist = [None, None, None, None, None]
                indicator = [None, None, None, None, None]
                for mind_name in memory.keys():
                    if mind_name == 'mg':
                        continue
                    if frame_id == 0:
                        memory_dist[mind_dict[mind_name]] = 0
                        indicator[mind_dict[mind_name]] = 0
                    else:
                        if frame_id%50 == 0:
                            memory_loc = obj_record[mind_name]['loc']
                        else:
                            memory_loc = memory[mind_name]['loc']
                        if memory_loc is not None:
                            curr_loc = np.array(curr_loc)
                            memory_loc = np.array(memory_loc)
                            # memory_dist[mind_dict[mind_name]] = int(np.linalg.norm(curr_loc - memory_loc) >0)
                            memory_dist[mind_dict[mind_name]] = np.linalg.norm(curr_loc - memory_loc)
                            indicator[mind_dict[mind_name]] = 1
                        else:
                            memory_dist[mind_dict[mind_name]] = 0
                            indicator[mind_dict[mind_name]] = 0
                # get predicted value
                memory_dist = np.array(memory_dist)
                indicator = np.array(indicator)

                event_input = np.hstack([p1_event, p2_event, memory_dist, indicator, output.reshape(-1)])

                event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))
                eid1 = np.argmax(p1_event)
                eid2 = np.argmax(p2_event)
                m = net(event_input)
                mind_pred = torch.softmax(m, dim = -1).data.cpu().numpy()
                mind_posterior = mind_pred
                # mind_posterior = get_posterior(mind_pred, event_mind_prob, mind_transition_prob, eid1, eid2)
                mind_output.append({'mind':mind_posterior, 'p1_event':p1_event, 'p2_event':p2_event})
                mid = np.argmax(mind_posterior)
                pred_combination = mind_combination[mid]

                memory = update_memory(memory, 'm1', pred_combination[0], curr_loc)
                memory = update_memory(memory, 'm2', pred_combination[1], curr_loc)
                memory = update_memory(memory, 'm12', pred_combination[2], curr_loc)
                memory = update_memory(memory, 'm21', pred_combination[3], curr_loc)
                memory = update_memory(memory, 'mc', pred_combination[4], curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

            assert len(mind_output) == len(mind_output_gt)
            if len(mind_output) > 0:
                with open(save_prefix + obj_name.split('/')[-1] + '.p', 'wb') as f:
                    pickle.dump([mind_output, mind_output_gt], f)

def get_gt_data_cnn():
    reannotation_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/annotations/'
    hog_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/feature_single/'
    att_path = '/home/shuwen/projects/six_minds/data/Six-MInds-Project/obj_oriented_event/att_vec/'
    model_path = '../mind_change_classifier/cptk_combined_raw_feature_mem/model_best.pth'
    save_path = './mind_cnn/'
    tree_path =  '../mind_change_classifier/BestTree_full_0915/'
    pair_feature_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/feature_pair/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(op.join('../mind_change_classifier/', 'distribution_record/mind_combination_dis.p'), 'rb') as f:
        event_mind_prob = pickle.load(f)

    for event in event_mind_prob.keys():
        event_prob = event_mind_prob[event]
        mind_count = np.zeros(1024)
        mind_combination = list(product([0, 1, 2, 3], repeat=5))
        for cid, combination in enumerate(mind_combination):
            if combination in event_prob:
                mind_count[cid] = event_prob[combination]
        mind_count = np.array(mind_count)
        mind_count = mind_count / float(np.sum(mind_count))
        assert abs(np.sum(mind_count) - 1) < 0.001
        event_mind_prob[event] = mind_count

    with open(op.join('../mind_change_classifier/', 'distribution_record/mind_transition_mat.p'), 'rb') as f:
        mind_transition_prob = pickle.load(f)

    clips = mind_test_clips
    # clips = clips_with_gt_event
    with open(op.join('../mind_change_classifier/', 'person_id.p'), 'rb') as f:
        person_ids = pickle.load(f)

    net = MLP_Feature_Mem()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    for clip in clips:
        clip_name = clip.split('.')[0]
        if not os.path.exists(reannotation_path + clip):
            continue
        with open(op.join(tree_path, clip), 'rb') as f:
            tree = pickle.load(f)
        with open(pair_feature_path + clip, 'rb') as f:
            pair_features = pickle.load(f)

        print(clip)
        save_prefix = save_path + clip.split('.')[0] + '/'
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)


        tracker_seg = tree['T']
        battery_seg = tree['B']

        tracker_events_by_frame = seg2frame_score(tracker_seg)
        battery_events_by_frame = seg2frame_score(battery_seg)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        # tracker_events_by_frame = reformat_events_gt(event_seg_tracker[clip_name],
        #                                              len(img_names))
        # battery_events_by_frame = reformat_events_gt(event_seg_battery[clip_name],
        #                                              len(img_names))
        assert  tracker_events_by_frame.shape == battery_events_by_frame.shape

        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(hog_path + clip, 'rb') as f:
            features = pickle.load(f)
        with open(att_path + clip, 'rb') as f:
            att_vec = pickle.load(f)


        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
            p1_hog = features[1]
            p2_hog = features[2]

        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
            p1_hog = features[2]
            p2_hog = features[1]

        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name","label"]
        obj_names = annt.name.unique()

        mind_dict = {'m1': 0, 'm2': 1, 'm12': 2, 'm21': 3, 'mc': 4}
        for obj_name in obj_names:
            # if os.path.exists(save_prefix + obj_name.split('/')[-1] + '.p'):
            #     continue
            if obj_name.startswith('P'):
                continue
            print(obj_name)

            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            mind_output_gt = []
            mind_output = []
            for frame_id in range(p1_events_by_frame.shape[0]):
                # curr_loc
                curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                curr_loc = get_grid_location(curr_df)

                obj_record = obj_records[obj_name][frame_id]
                mind_output_gt.append(obj_record)

                # event

                output = att_vec[obj_name][frame_id]
                p1_event = p1_events_by_frame[frame_id]
                p2_event = p2_events_by_frame[frame_id]

                # memory
                memory_dist = [None, None, None, None, None]
                indicator = [None, None, None, None, None]
                for mind_name in memory.keys():
                    if mind_name == 'mg':
                        continue
                    if frame_id == 0:
                        memory_dist[mind_dict[mind_name]] = 0
                        indicator[mind_dict[mind_name]] = 0
                    else:
                        if frame_id%50 == 0:
                            memory_loc = obj_record[mind_name]['loc']
                        else:
                            memory_loc = memory[mind_name]['loc']
                        if memory_loc is not None:
                            curr_loc = np.array(curr_loc)
                            memory_loc = np.array(memory_loc)
                            # memory_dist[mind_dict[mind_name]] = int(np.linalg.norm(curr_loc - memory_loc) >0)
                            memory_dist[mind_dict[mind_name]] = np.linalg.norm(curr_loc - memory_loc)
                            indicator[mind_dict[mind_name]] = 1
                        else:
                            memory_dist[mind_dict[mind_name]] = 0
                            indicator[mind_dict[mind_name]] = 0
                # get predicted value
                memory_dist = np.array(memory_dist)
                indicator = np.array(indicator)

                event_input = np.hstack([memory_dist, indicator])

                event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))

                hog_feature = np.hstack([p1_hog[frame_id], p2_hog[frame_id], pair_features[frame_id]])
                hog_feature = torch.from_numpy(hog_feature).float().cuda().view((1, -1))
                eid1 = np.argmax(p1_event)
                eid2 = np.argmax(p2_event)
                m = net(hog_feature, event_input)
                mind_pred = torch.softmax(m, dim = -1).data.cpu().numpy()
                mind_posterior = mind_pred
                # mind_posterior = get_posterior(mind_pred, event_mind_prob, mind_transition_prob, eid1, eid2)
                mind_output.append({'mind':mind_posterior, 'p1_event':p1_event, 'p2_event':p2_event})
                mid = np.argmax(mind_posterior)
                pred_combination = mind_combination[mid]

                memory = update_memory(memory, 'm1', pred_combination[0], curr_loc)
                memory = update_memory(memory, 'm2', pred_combination[1], curr_loc)
                memory = update_memory(memory, 'm12', pred_combination[2], curr_loc)
                memory = update_memory(memory, 'm21', pred_combination[3], curr_loc)
                memory = update_memory(memory, 'mc', pred_combination[4], curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

            assert len(mind_output) == len(mind_output_gt)
            if len(mind_output) > 0:
                with open(save_prefix + obj_name.split('/')[-1] + '.p', 'wb') as f:
                    pickle.dump([mind_output, mind_output_gt], f)

def get_gt_data_cnn_beam_search():
    reannotation_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/annotations/'
    hog_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/feature_single/'
    att_path = '/home/shuwen/projects/six_minds/data/Six-MInds-Project/obj_oriented_event/att_vec/'
    model_path = '../mind_change_classifier/cptk_combined_raw_feature_mem/model_best.pth'
    save_path = './mind_cnn_beam_search/'
    tree_path =  '../mind_change_classifier/BestTree_full_0915/'
    pair_feature_path = '/home/shuwen/projects/six_minds/data/data_preprocessing2/feature_pair/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(op.join('../mind_change_classifier/', 'distribution_record/mind_combination_dis.p'), 'rb') as f:
        event_mind_prob = pickle.load(f)

    for event in event_mind_prob.keys():
        event_prob = event_mind_prob[event]
        mind_count = np.zeros(1024)
        mind_combination = list(product([0, 1, 2, 3], repeat=5))
        for cid, combination in enumerate(mind_combination):
            if combination in event_prob:
                mind_count[cid] = event_prob[combination]
        mind_count = np.array(mind_count)
        mind_count = mind_count / float(np.sum(mind_count))
        assert abs(np.sum(mind_count) - 1) < 0.001
        event_mind_prob[event] = mind_count

    with open(op.join('../mind_change_classifier/', 'distribution_record/mind_transition_mat.p'), 'rb') as f:
        mind_transition_prob = pickle.load(f)

    clips = mind_test_clips
    # clips = clips_with_gt_event
    with open(op.join('../mind_change_classifier/', 'person_id.p'), 'rb') as f:
        person_ids = pickle.load(f)

    net = MLP_Feature_Mem()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    for clip in clips:
        clip_name = clip.split('.')[0]
        if not os.path.exists(reannotation_path + clip):
            continue
        with open(op.join(tree_path, clip), 'rb') as f:
            tree = pickle.load(f)
        with open(pair_feature_path + clip, 'rb') as f:
            pair_features = pickle.load(f)

        print(clip)
        save_prefix = save_path + clip.split('.')[0] + '/'
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)


        tracker_seg = tree['T']
        battery_seg = tree['B']

        tracker_events_by_frame = seg2frame_score(tracker_seg)
        battery_events_by_frame = seg2frame_score(battery_seg)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        # tracker_events_by_frame = reformat_events_gt(event_seg_tracker[clip_name],
        #                                              len(img_names))
        # battery_events_by_frame = reformat_events_gt(event_seg_battery[clip_name],
        #                                              len(img_names))
        assert  tracker_events_by_frame.shape == battery_events_by_frame.shape

        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(hog_path + clip, 'rb') as f:
            features = pickle.load(f)
        with open(att_path + clip, 'rb') as f:
            att_vec = pickle.load(f)


        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
            p1_hog = features[1]
            p2_hog = features[2]

        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
            p1_hog = features[2]
            p2_hog = features[1]

        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name","label"]
        obj_names = annt.name.unique()

        mind_dict = {'m1': 0, 'm2': 1, 'm12': 2, 'm21': 3, 'mc': 4}
        for obj_name in obj_names:
            # if os.path.exists(save_prefix + obj_name.split('/')[-1] + '.p'):
            #     continue
            if obj_name.startswith('P'):
                continue
            print(obj_name)

            memories = []
            for temp_id in range(5):
                memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                          'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
                memories.append(memory)
            mind_output_gt = []
            pred_minds = {0:[], 1:[], 2:[], 3:[], 4:[]}
            search_results = {}
            for frame_id in range(p1_events_by_frame.shape[0]):
                # curr_loc
                curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                curr_loc = get_grid_location(curr_df)

                obj_record = obj_records[obj_name][frame_id]
                mind_output_gt.append(obj_record)

                # memory
                all_possible_paths = np.array([1,2])
                all_possible_minds = np.array([1, 2])
                if frame_id == 0:
                    memory_dist = [None, None, None, None, None]
                    indicator = [None, None, None, None, None]
                    for mind_name in memories[0].keys():
                        if mind_name == 'mg':
                            continue
                        memory_dist[mind_dict[mind_name]] = 0
                        indicator[mind_dict[mind_name]] = 0

                    # get predicted value
                    memory_dist = np.array(memory_dist)
                    indicator = np.array(indicator)

                    event_input = np.hstack([memory_dist, indicator])

                    event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))

                    hog_feature = np.hstack([p1_hog[frame_id], p2_hog[frame_id], pair_features[frame_id]])
                    hog_feature = torch.from_numpy(hog_feature).float().cuda().view((1, -1))
                    m = net(hog_feature, event_input)
                    mind_pred = torch.softmax(m, dim=-1).data.cpu().numpy().reshape(-1)
                    new_score = -1*1e-6*np.ones(1024)
                    new_score[mind_pred > 0] = np.log(mind_pred[mind_pred > 0])
                    all_possible_paths = np.append(all_possible_paths, new_score)
                    all_possible_minds = np.append(all_possible_minds, -1*np.ones(1024))
                else:
                    for rid, result in enumerate(search_results[frame_id - 1]):
                        memory_dist = [None, None, None, None, None]
                        indicator = [None, None, None, None, None]
                        memory = result['memory']
                        score = result['score']
                        for mind_name in memory.keys():
                            if mind_name == 'mg':
                                continue
                            if frame_id == 0:
                                memory_dist[mind_dict[mind_name]] = 0
                                indicator[mind_dict[mind_name]] = 0
                            else:
                                if frame_id%50 == 0:
                                    memory_loc = obj_record[mind_name]['loc']
                                else:
                                    memory_loc = memory[mind_name]['loc']
                                if memory_loc is not None:
                                    curr_loc = np.array(curr_loc)
                                    memory_loc = np.array(memory_loc)
                                    # memory_dist[mind_dict[mind_name]] = int(np.linalg.norm(curr_loc - memory_loc) >0)
                                    memory_dist[mind_dict[mind_name]] = np.linalg.norm(curr_loc - memory_loc)
                                    indicator[mind_dict[mind_name]] = 1
                                else:
                                    memory_dist[mind_dict[mind_name]] = 0
                                    indicator[mind_dict[mind_name]] = 0
                        # get predicted value
                        memory_dist = np.array(memory_dist)
                        indicator = np.array(indicator)

                        event_input = np.hstack([memory_dist, indicator])

                        event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))

                        hog_feature = np.hstack([p1_hog[frame_id], p2_hog[frame_id], pair_features[frame_id]])
                        hog_feature = torch.from_numpy(hog_feature).float().cuda().view((1, -1))
                        m = net(hog_feature, event_input)
                        mind_pred = torch.softmax(m, dim = -1).data.cpu().numpy().reshape(-1)
                        new_score = mind_pred
                        new_score[mind_pred > 0] += np.log(mind_pred[mind_pred > 0])
                        new_score[mind_pred <= 0] -= 1e-6
                        all_possible_paths = np.append(all_possible_paths, new_score)
                        all_possible_minds = np.append(all_possible_minds, rid * np.ones(1024))

                all_possible_paths = all_possible_paths[2:]
                all_possible_minds = all_possible_minds[2:]
                mid = np.argsort(all_possible_paths)[::-1]
                # mind_posterior = mind_pred
                # mind_posterior = get_posterior(mind_pred, event_mind_prob, mind_transition_prob, eid1, eid2)
                # mind_output.append({'mind':mind_posterior, 'p1_event':p1_event, 'p2_event':p2_event})
                # mid = np.argsort(mind_posterior)[::-1]
                search_results[frame_id] = []
                pred_minds_temp = {0:[], 1:[], 2:[], 3:[], 4:[]}
                for temp_id in range(5):
                    com_id = mid[temp_id]
                    com_id_t = com_id%1024
                    pred_combination = mind_combination[com_id_t]
                    if frame_id == 0:
                        memory = copy.deepcopy(memories[0])
                    else:
                        rid = int(all_possible_minds[com_id])
                        memory = copy.deepcopy(search_results[frame_id - 1][rid]['memory'])
                    memory = update_memory(memory, 'm1', pred_combination[0], curr_loc)
                    memory = update_memory(memory, 'm2', pred_combination[1], curr_loc)
                    memory = update_memory(memory, 'm12', pred_combination[2], curr_loc)
                    memory = update_memory(memory, 'm21', pred_combination[3], curr_loc)
                    memory = update_memory(memory, 'mc', pred_combination[4], curr_loc)
                    memory = update_memory(memory, 'mg', 2, curr_loc)
                    temp_results = {}
                    temp_results['memory'] = memory
                    temp_results['score'] = all_possible_paths[com_id]
                    search_results[frame_id].append(temp_results)
                    if frame_id == 0:
                        pred_minds[temp_id].append(pred_combination)
                    else:
                        rid = int(all_possible_minds[com_id])
                        pred_minds_temp[temp_id] = copy.deepcopy(pred_minds[rid])
                        pred_minds_temp[temp_id].append(pred_combination)
                if frame_id > 0:
                    pred_minds = pred_minds_temp

            mind_output = pred_minds[0]
            assert len(mind_output) == len(mind_output_gt)
            if len(mind_output) > 0:
                with open(save_prefix + obj_name.split('/')[-1] + '.p', 'wb') as f:
                    pickle.dump([mind_output, mind_output_gt], f)

def test_raw_data():
    reannotation_path = './regenerate_annotation/'
    annotation_path = './reformat_annotation/'
    color_img_path = './annotations/'
    obj_bbox_path = './interpolate_bbox/'
    model_path = './cptk/model_best.pth'
    tree_path = './BestTree/BestTree_att_obj_id_w_raw_objs_01232021/'
    att_path = './att_vec_raw_obj_01202021/'
    save_path = './mind_raw_obj/'
    net = MLP_Combined_Att()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    with open('./person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = mind_test_clips
    mind_dict = {'m1': 0, 'm2': 1, 'm12': 2, 'm21': 3, 'mc': 4}
    mind_combination = list(product([0, 1, 2, 3], repeat=5))
    for clip in clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)

        with open(op.join(tree_path, clip), 'rb') as f:
            tree = pickle.load(f)
        with open(att_path + clip, 'rb') as f:
            att_vec = pickle.load(f)

        print(clip)
        save_prefix = save_path + clip.split('.')[0] + '/'
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)

        tracker_seg = tree['T']
        battery_seg = tree['B']
        tracker_events_by_frame = seg2frame_score(tracker_seg)
        battery_events_by_frame = seg2frame_score(battery_seg)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))

        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = glob.glob(obj_bbox_path + clip.split('.')[0] + '/*.p')
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            print(obj_name)
            with open(obj_name, 'rb') as f:
                obj_bboxs = pickle.load(f)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            mind_output_gt = []
            mind_output = []
            for frame_id in range(p1_events_by_frame.shape[0]):
                # curr_loc
                obj_bbox = obj_bboxs[frame_id]
                curr_loc = get_grid_location_using_bbox(obj_bbox)

                # gt
                gt_obj_name, gt_bbox = get_obj_name(obj_bbox, annt, frame_id)
                # img = cv2.imread(img_names[frame_id])

                if not gt_obj_name:
                    continue

                # cv2.rectangle(img, (int(obj_bbox[0]), int(obj_bbox[1])), (int(obj_bbox[2] + obj_bbox[0]), int(obj_bbox[3] + obj_bbox[1])), (255, 0, 0), thickness=2)
                # cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), thickness=2)
                # cv2.imshow('img', img)
                # cv2.waitKey(200)
                obj_record = obj_records[gt_obj_name][frame_id]
                mind_output_gt.append(obj_record)

                # event
                p1_event = p1_events_by_frame[frame_id]
                p2_event = p2_events_by_frame[frame_id]

                # memory
                memory_dist = [None, None, None, None, None]
                indicator = [None, None, None, None, None]
                for mind_name in memory.keys():
                    if mind_name == 'mg':
                        continue
                    if frame_id == 0:
                        memory_dist[mind_dict[mind_name]] = 0
                        indicator[mind_dict[mind_name]] = 0
                    else:
                        if frame_id % 50 == 0:
                            memory_loc = obj_record[mind_name]['loc']
                        else:
                            memory_loc = memory[mind_name]['loc']
                        if memory_loc is not None:
                            curr_loc = np.array(curr_loc)
                            memory_loc = np.array(memory_loc)
                            # memory_dist[mind_dict[mind_name]] = int(np.linalg.norm(curr_loc - memory_loc) >0)
                            memory_dist[mind_dict[mind_name]] = np.linalg.norm(curr_loc - memory_loc)
                            indicator[mind_dict[mind_name]] = 1
                        else:
                            memory_dist[mind_dict[mind_name]] = 0
                            indicator[mind_dict[mind_name]] = 0
                # get predicted value
                memory_dist = np.array(memory_dist)
                indicator = np.array(indicator)
                base_name = '/'.join(obj_name.split('/')[-2:])
                att_vec_key = './interpolate_bbox/' + base_name
                output = att_vec[att_vec_key][frame_id]
                event_input = np.hstack([p1_event, p2_event, memory_dist, indicator, output.reshape(-1)])

                event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))
                m = net(event_input)
                mind_pred = torch.softmax(m, dim=-1).data.cpu().numpy()
                mind_posterior = mind_pred
                # mind_posterior = get_posterior(mind_pred, event_mind_prob, mind_transition_prob, eid1, eid2)
                mind_output.append({'mind': mind_posterior, 'p1_event': p1_event, 'p2_event': p2_event})
                mid = np.argmax(mind_posterior)
                pred_combination = mind_combination[mid]
                memory = update_memory(memory, 'm1', pred_combination[0], curr_loc)
                memory = update_memory(memory, 'm2', pred_combination[1], curr_loc)
                memory = update_memory(memory, 'm12', pred_combination[2], curr_loc)
                memory = update_memory(memory, 'm21', pred_combination[3], curr_loc)
                memory = update_memory(memory, 'mc', pred_combination[4], curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

            assert len(mind_output) == len(mind_output_gt)
            if len(mind_output) > 0:
                with open(save_prefix + obj_name.split('/')[-1].split('.')[0] + '.p', 'wb') as f:
                    pickle.dump([mind_output, mind_output_gt], f)

def marginalize(predictions):
    assert abs(np.sum(predictions) - 1) < 0.0001
    mind_combination = list(product([0, 1, 2, 3], repeat=5))
    mind_dict = {key:predictions[0][mid] for mid, key in enumerate(mind_combination)}
    score_dict = {'m1':[0, 0, 0, 0], 'm2':[0, 0, 0, 0], 'm12':[0, 0, 0, 0], 'm21':[0, 0, 0, 0], 'mc':[0, 0, 0, 0]}
    key_hash = {0:'m1', 1:'m2', 2:'m12', 3:'m21', 4:'mc'}
    for mid, key in enumerate(mind_dict):
        for i in range(5):
            for j in range(4):
                if key[i] == j:
                    score_dict[key_hash[i]][j] += mind_dict[key]
    assert abs(sum(score_dict['m1']) - 1)< 0.0001
    assert abs(sum(score_dict['m2']) - 1)< 0.0001
    assert abs(sum(score_dict['m12']) - 1)< 0.0001
    assert abs(sum(score_dict['m21']) - 1)< 0.0001
    assert abs(sum(score_dict['mc']) - 1) < 0.0001

    return score_dict


def plot_confusion_matrix(cmc, title):
    df_cm = pd.DataFrame(cmc, range(cmc.shape[0]), range(cmc.shape[1]))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.savefig('./confusion_mat_plot/{}.png'.format(title))
    plt.show()

def roc_plot(mc_r, mc_s, title, type):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mc_r_one_hot = []
    for label in mc_r:
        label_t = [0, 0, 0, 0]
        label_t[label] = 1
        mc_r_one_hot.append(label_t)
    mc_r_one_hot = np.array(mc_r_one_hot)

    mc_s = np.array(mc_s)
    assert mc_s.shape == mc_r_one_hot.shape
    for i in range(4):
        fpr[i], tpr[i], _ = metrics.roc_curve(mc_r_one_hot[:, i], mc_s[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    plt.figure()
    d0 = {'fpr':fpr[0], 'tpr':tpr[0], 'belief dynamics':['class:' + str(0) + '(area = %0.2f)' % roc_auc[0]]*len(fpr[0])}
    df0 = pd.DataFrame(data=d0)
    d1 = {'fpr': fpr[1], 'tpr': tpr[1], 'belief dynamics': ['class:' + str(1) + '(area = %0.2f)' % roc_auc[1]]*len(fpr[1])}
    df1 = pd.DataFrame(data=d1)
    d2 = {'fpr': fpr[2], 'tpr': tpr[2], 'belief dynamics': ['class:' + str(2) + '(area = %0.2f)' % roc_auc[2]]*len(fpr[2])}
    df2 = pd.DataFrame(data=d2)
    d3 = {'fpr': fpr[3], 'tpr': tpr[3], 'belief dynamics': ['class:' + str(3) + '(area = %0.2f)' % roc_auc[3]]*len(fpr[3])}
    df3 = pd.DataFrame(data=d3)
    df = pd.concat([df0, df1, df2, df3])
    db = {'x':[0, 0], 'y':[1, 1]}
    db_f = pd.DataFrame(data=db)
    b = sn.lineplot(
        data=df, x="fpr", y="tpr",
        hue="belief dynamics", ci=None, legend="full", palette="Set2"
    )
    b.axes.set_title('Receiver operating characteristic:' + title, fontsize = 18)
    b.set_xlabel("Positive", fontsize=18)
    b.set_ylabel("True Positive Rate", fontsize=18)
    plt.setp(b.get_legend().get_texts(), fontsize=13)
    plt.setp(b.get_legend().get_title(), fontsize=13)
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.savefig('./roc_plot/{}_{}.png'.format(title, type))
    plt.show()
    # plt.figure()
    # lw = 2
    # colors = ['r', 'g', 'b', 'y']
    # for i in range(4):
    #     plt.plot(fpr[i], tpr[i], color=colors[i],
    #              lw=lw, label='class:' + str(i) + '(area = %0.2f)' % roc_auc[i])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate', fontsize = 18)
    # plt.ylabel('True Positive Rate', fontsize = 18)
    # plt.title('Receiver operating characteristic:' + title, fontsize = 18)
    # plt.legend(loc="lower right", fontsize = 13)
    # plt.savefig('./roc_plot/{}_{}.png'.format(title, type))
    # plt.show()

def result_eval():
    result_path = './mind_raw_obj/'
    type = 'mind_full'

    m1_r, m2_r, m12_r, m21_r, mc_r = [], [], [], [], []
    m1_p, m2_p, m12_p, m21_p, mc_p = [], [], [], [], []
    m1_s, m2_s, m12_s, m21_s, mc_s = [], [], [], [], []
    mind_combination = list(product([0, 1, 2, 3], repeat=5))
    for clip in clips_all:
        print(clip)
        clip_name = clip.split('.')[0]
        obj_files = glob.glob(result_path + clip_name + '/*.p')
        for obj_file in obj_files:
            with open(obj_file, 'rb') as f:
                outputs, gts = pickle.load(f)

            for fid, output in enumerate(outputs):
                gt = gts[fid]

                mid = np.argmax(output['mind'])
                pred = mind_combination[mid]
                # score_dict = marginalize(output['mind'])
                # m1_s.append(score_dict['m1'])
                # m2_s.append(score_dict['m2'])
                # m12_s.append(score_dict['m12'])
                # m21_s.append(score_dict['m21'])
                # mc_s.append(score_dict['mc'])

                m1_p.append(pred[0])
                m2_p.append(pred[1])
                m12_p.append(pred[2])
                m21_p.append(pred[3])
                mc_p.append(pred[4])

                m1_r.append(gt['m1']['fluent'])
                m2_r.append(gt['m2']['fluent'])
                m12_r.append(gt['m12']['fluent'])
                m21_r.append(gt['m21']['fluent'])
                mc_r.append(gt['mc']['fluent'])

    results_mc = metrics.classification_report(mc_r, mc_p, digits=3)
    results_m1 = metrics.classification_report(m1_r, m1_p, digits=3)
    results_m2 = metrics.classification_report(m2_r, m2_p, digits=3)
    results_m12 = metrics.classification_report(m12_r, m12_p, digits=3)
    results_m21 = metrics.classification_report(m21_r, m21_p, digits=3)
    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    # cmc = metrics.confusion_matrix(mc_r, mc_p)
    # cm1 = metrics.confusion_matrix(m1_r, m1_p)
    # cm2 = metrics.confusion_matrix(m2_r, m2_p)
    # cm12 = metrics.confusion_matrix(m12_r, m12_p)
    # cm21 = metrics.confusion_matrix(m21_r, m21_p)
    # plot_confusion_matrix(cmc, 'mc')
    # plot_confusion_matrix(cm1, 'm1')
    # plot_confusion_matrix(cm2, 'm2')
    # plot_confusion_matrix(cm12, 'm12')
    # plot_confusion_matrix(cm21, 'm21')

    # results_mc = metrics.roc_auc_score(mc_r, mc_s)
    # results_m1 = metrics.roc_auc_score(m1_r, m1_s)
    # results_m2 = metrics.roc_auc_score(m2_r, m2_s)
    # results_m12 = metrics.roc_auc_score(m12_r, m12_s)
    # results_m21 = metrics.roc_auc_score(m21_r, m21_s)
    # with open('./roc_socre.p', 'wb') as f:
    #     pickle.dump([[mc_r, m1_r, m2_r, m12_r, m21_r], [mc_s, m1_s, m2_s, m12_s, m21_s]], f)
    # roc_plot(mc_r, mc_s, 'mc', type)
    # roc_plot(m1_r, m1_s, 'm1', type)
    # roc_plot(m2_r, m2_s, 'm2', type)
    # roc_plot(m12_r, m12_s, 'm12', type)
    # roc_plot(m21_r, m21_s, 'm21', type)

def result_eval_cnn_beam_search():
    result_path = './mind_cnn_beam_search/'
    type = 'mind_full'

    m1_r, m2_r, m12_r, m21_r, mc_r = [], [], [], [], []
    m1_p, m2_p, m12_p, m21_p, mc_p = [], [], [], [], []
    m1_s, m2_s, m12_s, m21_s, mc_s = [], [], [], [], []
    mind_combination = list(product([0, 1, 2, 3], repeat=5))
    for clip in clips_all:
        print(clip)
        clip_name = clip.split('.')[0]
        obj_files = glob.glob(result_path + clip_name + '/*.p')
        for obj_file in obj_files:
            with open(obj_file, 'rb') as f:
                outputs, gts = pickle.load(f)

            for fid, output in enumerate(outputs):
                gt = gts[fid]
                pred = output
                # score_dict = marginalize(output['mind'])
                # m1_s.append(score_dict['m1'])
                # m2_s.append(score_dict['m2'])
                # m12_s.append(score_dict['m12'])
                # m21_s.append(score_dict['m21'])
                # mc_s.append(score_dict['mc'])

                m1_p.append(pred[0])
                m2_p.append(pred[1])
                m12_p.append(pred[2])
                m21_p.append(pred[3])
                mc_p.append(pred[4])

                m1_r.append(gt['m1']['fluent'])
                m2_r.append(gt['m2']['fluent'])
                m12_r.append(gt['m12']['fluent'])
                m21_r.append(gt['m21']['fluent'])
                mc_r.append(gt['mc']['fluent'])

    results_mc = metrics.classification_report(mc_r, mc_p, digits=3)
    results_m1 = metrics.classification_report(m1_r, m1_p, digits=3)
    results_m2 = metrics.classification_report(m2_r, m2_p, digits=3)
    results_m12 = metrics.classification_report(m12_r, m12_p, digits=3)
    results_m21 = metrics.classification_report(m21_r, m21_p, digits=3)
    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    # cmc = metrics.confusion_matrix(mc_r, mc_p)
    # cm1 = metrics.confusion_matrix(m1_r, m1_p)
    # cm2 = metrics.confusion_matrix(m2_r, m2_p)
    # cm12 = metrics.confusion_matrix(m12_r, m12_p)
    # cm21 = metrics.confusion_matrix(m21_r, m21_p)
    # plot_confusion_matrix(cmc, 'mc')
    # plot_confusion_matrix(cm1, 'm1')
    # plot_confusion_matrix(cm2, 'm2')
    # plot_confusion_matrix(cm12, 'm12')
    # plot_confusion_matrix(cm21, 'm21')

    # results_mc = metrics.roc_auc_score(mc_r, mc_s)
    # results_m1 = metrics.roc_auc_score(m1_r, m1_s)
    # results_m2 = metrics.roc_auc_score(m2_r, m2_s)
    # results_m12 = metrics.roc_auc_score(m12_r, m12_s)
    # results_m21 = metrics.roc_auc_score(m21_r, m21_s)
    # with open('./roc_socre.p', 'wb') as f:
    #     pickle.dump([[mc_r, m1_r, m2_r, m12_r, m21_r], [mc_s, m1_s, m2_s, m12_s, m21_s]], f)
    # roc_plot(mc_r, mc_s, 'mc', type)
    # roc_plot(m1_r, m1_s, 'm1', type)
    # roc_plot(m2_r, m2_s, 'm2', type)
    # roc_plot(m12_r, m12_s, 'm12', type)
    # roc_plot(m21_r, m21_s, 'm21', type)

def save_roc_plot():
    with open('./roc_socre.p', 'rb') as f:
        [[mc_r, m1_r, m2_r, m12_r, m21_r], [mc_s, m1_s, m2_s, m12_s, m21_s]] = pickle.load(f)
    type = 'mind_full'
    roc_plot(mc_r, mc_s, 'mc', type)
    roc_plot(m1_r, m1_s, 'm1', type)
    roc_plot(m2_r, m2_s, 'm2', type)
    roc_plot(m12_r, m12_s, 'm12', type)
    roc_plot(m21_r, m21_s, 'm21', type)


if __name__ == '__main__':
    test_raw_data()
    get_gt_data()
    result_eval()
    # save_roc_plot()