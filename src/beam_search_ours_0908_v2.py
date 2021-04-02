import glob
import sys
from metadata import *
from utils import *
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
import argparse
import pickle
import numpy, scipy.io
import torch
import numpy as np
import joblib
import pywt
from overall_event_get_input import *
import joblib
from Atomic_node_only_lstm import Atomic_node_only_lstm
import copy
import time
import threading
import os, os.path
from metric import *
# from BeamSearchClass import *
from multiprocessing import Process
#import torch.multiprocessing
from  Atomic_node_only_lstm_517 import  Atomic_node_only_lstm_first_view
# torch.multiprocessing.set_start_method('spawn', force='True')
from joblib import Parallel, delayed
from os import listdir
from para_bank import generate_para_bank
from itertools import product


class BeamSearch(object):

    # ================
    # init functions
    # ================
    def __init__(self, args, event_net, atomic_net):

        self.args = args
        self.epsilon = 0.000001
        self.init_cps_all = pickle.load(open(args.init_cps, 'rb'), encoding='latin1')

        self.event_net = event_net
        self.atomic_net = atomic_net
        self.event_trans_table = pickle.load(open(os.path.join(args.stat_path, 'event_transition_normed.p'), 'rb'),
                                             encoding='latin1')

    def init(self, clip):

        self.clip = clip  # with .p

        self.clip_length = event_seg_tracker[clip.split('.')[0]][-1][1] + 1

        # initial cps
        self.init_cps_T = self.init_cps_all[0][self.clip]
        self.init_cps_B = self.init_cps_all[1][self.clip]
        # add the final frame
        self.init_cps_T.append(self.clip_length)
        self.init_cps_T = list(np.unique(self.init_cps_T))
        self.init_cps_B.append(self.clip_length)
        self.init_cps_B = list(np.unique(self.init_cps_B))

        # with open(self.args.tracker_bbox + clip, 'rb') as f:
        #     self.person_tracker_bbox = pickle.load(f, encoding='latin1')
        # with open(self.args.battery_bbox + clip, 'rb') as f:
        #     self.person_battery_bbox = pickle.load(f, encoding='latin1')

        # attmat
        # with open(self.args.attmat_path + clip, 'rb') as f:
        #     self.attmat_obj = pickle.load(f, encoding='latin1')

        # with open(self.args.cate_path + clip, 'rb') as f:
        #     self.category = pickle.load(f, encoding='latin1')
        #
        # # feature
        with open(os.path.join(self.args.data_path2, 'feature_single', clip), 'rb') as f:
            self.feature_single = pickle.load(f, encoding='latin1')

        if tracker_skeID[clip.split('.')[0]] == 'skele1.p':
            self.T_skeID = 1
            self.B_skeID = 2
        elif tracker_skeID[clip.split('.')[0]] == 'skele2.p':
            self.T_skeID = 2
            self.B_skeID = 1

        # init root node and tree for each new clip
        self.root_node = {'T': {'cp': [0], 'event': [], 'mind': []}, 'B': {'cp': [0], 'event': [], 'mind': []},
                          'score': {'prior_aggr': {'cnt_joint': 0., 'cnt_single': 0.},
                                    'prior_event': {'e_T': 0., 'cnt_T': 0., 'e_S': 0., 'cnt_S': 0., 'T_last_type': None,
                                                    'B_last_type': None},
                                    'prior_fluent': 0.,
                                    'like_P': {'T_D_i': 0., 'T_CNT_i': 0., 'B_D_i': 0., 'B_CNT_i': 0., 'T_D_t': 0.,
                                               'T_CNT_t': 0., 'B_D_t': 0., 'B_CNT_t': 0., 'last_T_hist': None,
                                               'last_B_hist': None, 'D_s': 0., 'CNT_s': 0.},
                                    'like_E': {'e_E': 0., 'CNT': 0.},
                                    'like_M': 0,
                                    'T_update_flag': 0,
                                    'B_update_flag': 0
                                    }
                          }
        self.Tree = {'nodes': [self.root_node], 'level': 0}

        self.check_event_flag = True

    # ================
    # utils functions
    # ================
    def seg2frame(self, node):
        segs = node['cp']
        events = node['event']

        frame_labels = np.zeros((segs[-1]))
        for seg_id, seg in enumerate(segs[:-1]):
            event = events[seg_id][0]
            start = seg
            end = segs[seg_id + 1]
            frame_labels[start: end] = event

        return frame_labels

    def check_event(self, node):
        tracker_frames = self.seg2frame(node['T'])
        battery_frames = self.seg2frame(node['B'])
        overlap_id = min(tracker_frames.shape[0], battery_frames.shape[0])
        compare = np.abs(tracker_frames[:overlap_id] - battery_frames[:overlap_id])
        return np.sum(compare) > 0

    def cps2segs(self, cps):
        segs = []
        if len(cps) >= 2:
            cp_l = cps[0]
            segs = []
            for idx in range(1, len(cps)):
                cp_r = cps[idx]
                segs.append([cp_l, cp_r])
                cp_l = cp_r
        return segs

    def combcps(self, cps1, cps2):
        comb = []
        comb.extend(cps1)
        comb.extend(cps2)
        comb = list(np.sort(np.unique(np.array(comb))))

        return comb

    def seg2cp(self, segs):
        cp = []
        for seg in segs:
            cp.append(seg[0])
        return cp

    def ori_seg_id(self, ori_cps, seg):
        if len(ori_cps) > 1:
            for idx in range(1, len(ori_cps)):
                if seg[0] >= ori_cps[idx - 1] and seg[1] <= ori_cps[idx]:
                    return idx - 1
        return None

    def freq2hist(self, freq_feature):
        seg_features = np.empty((1, 0))
        for dim_id in range(freq_feature.shape[1]):
            video_vec = freq_feature[:, dim_id] / 4
            hist, bin_edges = np.histogram(video_vec, bins=self.args.hist_bin, density=True)
            seg_features = np.hstack([seg_features, hist.reshape((1, -1))])
        return seg_features

    def temporal2freq(self, feature):
        coeffs = pywt.dwt(feature, 'sym2')
        cA1, _ = coeffs
        new_feature = self.freq2hist(cA1)
        return new_feature

    def find_sep_gt(self, tracker_gt_seg, segs_T):
        curr_gt_seg = []
        start_frame = segs_T[0][0]
        end_frame = segs_T[-1][1]
        for seg in tracker_gt_seg:
            max_start = max(seg[0], start_frame)
            min_end = min(seg[1], end_frame)
            if max_start < min_end:
                curr_gt_seg.append(seg)
        return curr_gt_seg

    # ================
    # score functions
    # ================
    def prior_energy_aggr(self, node):

        N_p = len(node['T']['cp']) / float(node['T']['cp'][-1]) + len(node['B']['cp']) / float(node['B']['cp'][-1])

        e_aggr = self.args.lambda_1 * N_p

        return e_aggr, node

    def prior_energy_event(self, node):
        # event validness score (temporal transition and spatial concurrency, from dataset)
        e_T = node['score']['prior_event']['e_T']
        cnt_T = node['score']['prior_event']['cnt_T']
        T_last_type = node['score']['prior_event']['T_last_type']
        B_last_type = node['score']['prior_event']['B_last_type']

        # temporal transition
        T_new_type = node['T']['event'][-1][0]
        if T_last_type is not None:
            trans_key = (T_last_type, T_new_type)
            e_T += np.log(self.event_trans_table[trans_key])
            cnt_T += 1
        T_last_type = T_new_type

        B_new_type = node['B']['event'][-1][0]
        if B_last_type is not None:
            trans_key = (B_last_type, B_new_type)
            e_T += np.log(self.event_trans_table[trans_key])
            cnt_T += 1
        B_last_type = B_new_type

        node['score']['prior_event']['e_T'] = e_T
        node['score']['prior_event']['cnt_T'] = cnt_T
        node['score']['prior_event']['T_last_type'] = T_last_type
        node['score']['prior_event']['B_last_type'] = B_last_type

        e_T = e_T / (cnt_T + self.epsilon)

        # spatial concurrency
        e_S = node['score']['prior_event']['e_S']
        cnt_S = node['score']['prior_event']['cnt_S']

        segs_T = self.cps2segs(node['T']['cp'])
        segs_B = self.cps2segs(node['B']['cp'])

        if node['score']['T_update_flag'] == 1:
            for idx2 in range(len(segs_B)):
                if segs_T[-1][0] >= segs_B[idx2][1]:
                    continue
                elif segs_T[-1][1] <= segs_B[idx2][0]:
                    break
                else:
                    event_T = node['T']['event'][-1][0]
                    event_B = node['B']['event'][idx2][0]
                    if event_T == event_B:
                        e_S += np.log(1)
                        cnt_S += 1
                    else:
                        e_S += np.log(0 + self.epsilon)
                        cnt_S += 1

        if node['score']['B_update_flag'] == 1:
            for idx1 in range(len(segs_T)):
                if segs_T[idx1][0] >= segs_B[-1][1]:
                    break
                elif segs_T[idx1][1] <= segs_B[-1][0]:
                    continue
                else:
                    event_T = node['T']['event'][idx1][0]
                    event_B = node['B']['event'][-1][0]
                    if event_T == event_B:
                        e_S += np.log(1)
                        cnt_S += 1
                    else:
                        e_S += np.log(0 + self.epsilon)
                        cnt_S += 1
        if node['score']['T_update_flag'] == 1 and node['score']['B_update_flag'] == 1:
            event_T = node['T']['event'][-1][0]
            event_B = node['B']['event'][-1][0]
            if event_T == event_B:
                e_S -= np.log(1)
                cnt_S -= 1
            else:
                e_S -= np.log(0 + self.epsilon)
                cnt_S -= 1

        node['score']['prior_event']['e_S'] = e_S
        node['score']['prior_event']['cnt_S'] = cnt_S

        e_S = e_S / (cnt_S + self.epsilon)

        e_event = -self.args.lambda_2 * e_T - self.args.lambda_3 * e_S

        return e_event, node

    def likelihood_energy_P(self, node):

        # tracker
        # inner particle distance
        T_D_i = node['score']['like_P']['T_D_i']
        T_CNT_i = node['score']['like_P']['T_CNT_i']

        seg = [node['T']['cp'][-2], node['T']['cp'][-1]]
        feature = self.feature_single[self.T_skeID][seg[0]:seg[1], :]
        sum_tmp = 0
        for idx in range(1, feature.shape[0]):
            sum_tmp += np.linalg.norm(feature[idx - 1] - feature[idx])
        sum_tmp = sum_tmp / float(feature.shape[0])
        T_D_i += sum_tmp
        T_CNT_i += 1

        node['score']['like_P']['T_D_i'] = T_D_i
        node['score']['like_P']['T_CNT_i'] = T_CNT_i

        # inter particle distance - T
        T_D_t = node['score']['like_P']['T_D_t']
        T_CNT_t = node['score']['like_P']['T_CNT_t']
        last_T_hist = node['score']['like_P']['last_T_hist']
        T_hist = self.temporal2freq(feature)
        if last_T_hist is not None:
            T_D_t += np.linalg.norm(last_T_hist - T_hist) / 312.
            T_CNT_t += 1

        node['score']['like_P']['T_D_t'] = T_D_t
        node['score']['like_P']['T_CNT_t'] = T_CNT_t
        node['score']['like_P']['last_T_hist'] = T_hist

        # battery
        # inner particle distance
        B_D_i = node['score']['like_P']['B_D_i']
        B_CNT_i = node['score']['like_P']['B_CNT_i']
        seg = [node['B']['cp'][-2], node['B']['cp'][-1]]
        feature = self.feature_single[self.B_skeID][seg[0]:seg[1], :]
        sum_tmp = 0
        for idx in range(1, feature.shape[0]):
            sum_tmp += np.linalg.norm(feature[idx - 1] - feature[idx])
        sum_tmp = sum_tmp / float(feature.shape[0])
        B_D_i += sum_tmp
        B_CNT_i += 1

        node['score']['like_P']['B_D_i'] = B_D_i
        node['score']['like_P']['B_CNT_i'] = B_CNT_i

        # inter particle distance - T
        B_D_t = node['score']['like_P']['B_D_t']
        B_CNT_t = node['score']['like_P']['B_CNT_t']
        last_B_hist = node['score']['like_P']['last_B_hist']
        B_hist = self.temporal2freq(feature)
        if last_B_hist is not None:
            B_D_t += np.linalg.norm(last_B_hist - B_hist) / 312.
            B_CNT_t += 1

        node['score']['like_P']['B_D_t'] = B_D_t
        node['score']['like_P']['B_CNT_t'] = B_CNT_t
        node['score']['like_P']['last_B_hist'] = B_hist

        # inter particle  distance - S
        D_s = node['score']['like_P']['D_s']
        CNT_s = node['score']['like_P']['CNT_s']

        if node['score']['T_update_flag'] == 1 and node['score']['B_update_flag'] == 1:
            # tracker
            new_segs_T = self.cps2segs(self.combcps(node['T']['cp'][-2:], node['B']['cp']))
            for seg in new_segs_T:
                ori_idx_T = self.ori_seg_id(node['T']['cp'][-2:], seg)
                ori_idx_B = self.ori_seg_id(node['B']['cp'], seg)

                if ori_idx_T is None or ori_idx_B is None or node['T']['event'][-1][0] != 0 or \
                        node['B']['event'][ori_idx_B][0] != 0:
                    continue
                else:
                    hist_T = self.temporal2freq(self.feature_single[self.T_skeID][seg[0]:seg[1], :])
                    hist_B = self.temporal2freq(self.feature_single[self.B_skeID][seg[0]:seg[1], :])
                    D_s += np.linalg.norm(hist_T - hist_B) / 312.
                    CNT_s += 1
            # battery
            new_segs_B = self.cps2segs(self.combcps(node['B']['cp'][-2:], node['T']['cp']))
            for seg in new_segs_B:
                ori_idx_T = self.ori_seg_id(node['T']['cp'], seg)
                ori_idx_B = self.ori_seg_id(node['B']['cp'][-2:], seg)

                if ori_idx_T is None or ori_idx_B is None or node['T']['event'][ori_idx_T][0] != 0 or \
                        node['B']['event'][-1][0] != 0:
                    continue
                else:
                    hist_T = self.temporal2freq(self.feature_single[self.T_skeID][seg[0]:seg[1], :])
                    hist_B = self.temporal2freq(self.feature_single[self.B_skeID][seg[0]:seg[1], :])
                    D_s += np.linalg.norm(hist_T - hist_B) / 312.
                    CNT_s += 1
            # comb
            new_segs_comb = self.cps2segs(self.combcps(node['B']['cp'][-2:], node['T']['cp'][-2:]))
            for seg in new_segs_comb:
                ori_idx_T = self.ori_seg_id(node['T']['cp'][-2:], seg)
                ori_idx_B = self.ori_seg_id(node['B']['cp'][-2:], seg)

                if ori_idx_T is None or ori_idx_B is None or node['T']['event'][-1][0] != 0 or \
                        node['B']['event'][-1][0] != 0:
                    continue
                else:
                    hist_T = self.temporal2freq(self.feature_single[self.T_skeID][seg[0]:seg[1], :])
                    hist_B = self.temporal2freq(self.feature_single[self.B_skeID][seg[0]:seg[1], :])
                    D_s -= np.linalg.norm(hist_T - hist_B) / 312.
                    CNT_s -= 1
        elif node['score']['T_update_flag'] == 1 and node['score']['B_update_flag'] == 0:
            # tracker
            new_segs_T = self.cps2segs(self.combcps(node['T']['cp'][-2:], node['B']['cp']))
            for seg in new_segs_T:
                ori_idx_T = self.ori_seg_id(node['T']['cp'][-2:], seg)
                ori_idx_B = self.ori_seg_id(node['B']['cp'], seg)
                if ori_idx_T is None or ori_idx_B is None or node['T']['event'][-1][0] != 0 or \
                        node['B']['event'][ori_idx_B][0] != 0:
                    continue
                else:
                    hist_T = self.temporal2freq(self.feature_single[self.T_skeID][seg[0]:seg[1], :])
                    hist_B = self.temporal2freq(self.feature_single[self.B_skeID][seg[0]:seg[1], :])
                    D_s += np.linalg.norm(hist_T - hist_B) / 312.
                    CNT_s += 1
        elif node['score']['T_update_flag'] == 0 and node['score']['B_update_flag'] == 1:
            # battery
            new_segs_B = self.cps2segs(self.combcps(node['B']['cp'][-2:], node['T']['cp']))
            for seg in new_segs_B:
                ori_idx_T = self.ori_seg_id(node['T']['cp'], seg)
                ori_idx_B = self.ori_seg_id(node['B']['cp'][-2:], seg)

                if ori_idx_T is None or ori_idx_B is None or node['T']['event'][ori_idx_T][0] != 0 or \
                        node['B']['event'][-1][0] != 0:
                    continue
                else:
                    hist_T = self.temporal2freq(self.feature_single[self.T_skeID][seg[0]:seg[1], :])
                    hist_B = self.temporal2freq(self.feature_single[self.B_skeID][seg[0]:seg[1], :])
                    D_s += np.linalg.norm(hist_T - hist_B) / 312.
                    CNT_s += 1

        node['score']['like_P']['D_s'] = D_s
        node['score']['like_P']['CNT_s'] = CNT_s

        # sum
        e_P = self.args.lambda_5 * (
                    T_D_i / (T_CNT_i + self.epsilon) + B_D_i / (B_CNT_i + self.epsilon)) - self.args.lambda_6 * (
                          D_s / (CNT_s + self.epsilon)) \
              - self.args.lambda_7 * (T_D_t / (T_CNT_t + self.epsilon) + B_D_t / (B_CNT_t + self.epsilon))

        # e_P = self.args.beta_1 * (T_D_i / (T_CNT_i + self.epsilon) + B_D_i / (B_CNT_i + self.epsilon)) - self.args.beta_3 * (T_D_t / (T_CNT_t + self.epsilon) + B_D_t / (B_CNT_t + self.epsilon))

        return e_P, node

    def likelihood_energy_E(self, node):  # todo: add the event validness check here

        e_E = node['score']['like_E']['e_E']
        CNT = node['score']['like_E']['CNT']

        if node['score']['T_update_flag'] == 1:
            e_E += np.log(node['T']['event'][-1][1])
            CNT += 1
        if node['score']['B_update_flag'] == 1:
            e_E += np.log(node['B']['event'][-1][1])
            CNT += 1

        node['score']['like_E']['e_E'] = e_E
        node['score']['like_E']['CNT'] = CNT

        e_E = e_E / CNT
        energy_E = -self.args.lambda_8 * e_E

        return energy_E, node

    # ================
    # tree functions
    # ================
    def tree_prune(self, all_possible_path):
        '''
        input:
            all possible paths in this level
        :return:
            top N paths
        '''

        score_all = []
        node_ids = []
        all_possible_nodes_new = []

        for idx, node in enumerate(all_possible_path):
            # calculate score for the current node/path
            e_aggr, node = self.prior_energy_aggr(node)
            e_event, node = self.prior_energy_event(node)
            e_P, node = self.likelihood_energy_P(node)
            e_E, node = self.likelihood_energy_E(node)

            node_score = -e_aggr - e_event - e_P - e_E
            score_all.append(node_score)
            node_ids.append(idx)
            all_possible_nodes_new.append(node)

        ordered_index = list(np.argsort(np.array(score_all))[::-1])
        selected_index = ordered_index[:self.args.topN]
        node_ids = np.array(node_ids)
        top_node_ids = node_ids[selected_index]

        self.Tree['nodes'] = []
        for node_id in top_node_ids:
            node = all_possible_nodes_new[node_id]
            self.Tree['nodes'].append(node)

    def node_expand(self, parent_node):
        '''
        input:
            parent checkpoint node
        :return:
            all possible children nodes
        '''
        t_node_id = parent_node['T']['cp'][-1]
        possible_t_cp = []
        for i in range(len(self.init_cps_T)):
            if self.init_cps_T[i] > t_node_id:
                possible_t_cp.append(self.init_cps_T[i])

        b_node_id = parent_node['B']['cp'][-1]
        possible_b_cp = []
        for j in range(len(self.init_cps_B)):
            if self.init_cps_B[j] > b_node_id:
                possible_b_cp.append(self.init_cps_B[j])

        return possible_t_cp, possible_b_cp

    def tree_grow(self):

        '''
        input:
            current top N possible path
        :return:
            new top N possible path
        '''
        all_possible_nodes = []
        start_time = time.time()
        for idx, parent_node in enumerate(self.Tree['nodes']):

            possible_t_cp, possible_b_cp = self.node_expand(parent_node)
            search_N_cp_T = min(len(possible_t_cp), self.args.search_N_cp)
            search_N_cp_B = min(len(possible_b_cp), self.args.search_N_cp)

            # all possible paths
            if len(possible_t_cp) >= 1 and len(possible_b_cp) >= 1:
                for cp_t_id in possible_t_cp[:search_N_cp_T]:
                    for cp_b_id in possible_b_cp[:search_N_cp_B]:

                        # T
                        new_node = copy.deepcopy(parent_node)
                        new_node['T']['cp'].append(cp_t_id)
                        start = parent_node['T']['cp'][-1]
                        end = cp_t_id
                        try:
                            with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'tracker',
                                              '{}_{}.p'.format(start, end)), 'rb') as f:
                                outputs = pickle.load(f)
                        except:
                            continue

                        new_node['T']['event'].append([np.argmax(outputs[0]), np.max(outputs[0])])

                        # B
                        new_node['B']['cp'].append(cp_b_id)
                        start = parent_node['B']['cp'][-1]
                        end = cp_b_id

                        try:
                            with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'battery',
                                              '{}_{}.p'.format(start, end)), 'rb') as f:
                                outputs = pickle.load(f)
                        except:
                            continue

                        new_node['B']['event'].append([np.argmax(outputs[0]), np.max(outputs[0])])

                        new_node['score']['T_update_flag'] = 1
                        new_node['score']['B_update_flag'] = 1

                        all_possible_nodes.append(new_node)

            elif len(possible_t_cp) == 0 and len(possible_b_cp) >= 1:
                for cp_b_id in possible_b_cp[:search_N_cp_B]:

                    new_node1 = copy.deepcopy(parent_node)
                    # add new cp
                    new_node1['B']['cp'].append(cp_b_id)

                    # predict event for current seg
                    # battery
                    start = parent_node['B']['cp'][-1]
                    end = cp_b_id
                    try:
                        with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'battery',
                                          '{}_{}.p'.format(start, end)), 'rb') as f:
                            outputs = pickle.load(f)
                    except:
                        continue

                    new_node1['B']['event'].append([np.argmax(outputs[0]), np.max(outputs[0])])

                    new_node1['score']['T_update_flag'] = 0
                    new_node1['score']['B_update_flag'] = 1

                    all_possible_nodes.append(new_node1)

            elif len(possible_t_cp) >= 1 and len(possible_b_cp) == 0:

                for cp_t_id in possible_t_cp[:search_N_cp_T]:

                    new_node1 = copy.deepcopy(parent_node)
                    # add new cp
                    new_node1['T']['cp'].append(cp_t_id)
                    # predict event for current seg
                    # battery
                    start = parent_node['T']['cp'][-1]
                    end = cp_t_id
                    try:
                        with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'tracker',
                                          '{}_{}.p'.format(start, end)), 'rb') as f:
                            outputs = pickle.load(f)
                    except:
                        continue

                    new_node1['T']['event'].append([np.argmax(outputs[0]), np.max(outputs[0])])
                    new_node1['score']['T_update_flag'] = 1
                    new_node1['score']['B_update_flag'] = 0
                    all_possible_nodes.append(new_node1)

            else:

                all_possible_nodes.append(copy.deepcopy(parent_node))

        if len(all_possible_nodes) == len(self.Tree['nodes']):
            return self.Tree['nodes'][0]
        else:
            flag = self.tree_prune(all_possible_nodes)
            self.Tree['level'] += 1

def finetune_para(atomic_net, event_net, clip_idx, clip, para_idx, para, args, clip_len, para_len):

    pid = threading.get_ident()
    print('pid: {} clip {}/{} para {}/{}'.format(pid, clip_idx, clip_len, para_idx, para_len))
    beam_search = BeamSearch(args, event_net, atomic_net)

    args.topN = para['topN']
    args.lambda_1 = para['lambda_1']
    args.lambda_2 = para['lambda_2']
    args.lambda_3 = para['lambda_3']
    args.lambda_5 = para['lambda_5']
    args.lambda_6 = para['lambda_6']
    args.lambda_7 = para['lambda_7']
    args.lambda_8 = para['lambda_8']

    args.search_N_cp = para['search_N_cp']

    beam_search.init(clip)
    while True:
        Tree_best=beam_search.tree_grow()
        if Tree_best is not None:
            break

    # evaluation
    cps_T=Tree_best['T']['cp']
    cps_B=Tree_best['B']['cp']
    event_T=Tree_best['T']['event']
    event_B=Tree_best['B']['event']

    # ---------- seg
    segs_T=beam_search.cps2segs(cps_T)
    segs_B=beam_search.cps2segs(cps_B)
    tracker_gt_seg = event_seg_tracker[clip.split('.')[0]]
    battery_gt_seg = event_seg_battery[clip.split('.')[0]]
    err_seg = segment_error(segs_T, tracker_gt_seg, args.seg_alpha) + segment_error(segs_B, battery_gt_seg,args.seg_alpha)

    # ------------- event
    # tracker
    event_gt_T = event_seg_tracker[clip.split('.')[0]]
    len_T = event_gt_T[-1][1]
    frame_events_T_gt = np.zeros((len_T))
    for i, seg in enumerate(event_gt_T):
        start = event_gt_T[i][0]
        end = event_gt_T[i][1]
        event = event_gt_T[i][2]
        frame_events_T_gt[start:end] = event

    frame_events_T = np.zeros((len_T))
    for i, seg in enumerate(segs_T):
        event = event_T[i][0]
        start = seg[0]
        end = seg[1]
        frame_events_T[start:end] = event

    # battery
    event_gt_B = event_seg_battery[clip.split('.')[0]]
    len_B = event_gt_B[-1][1]
    frame_events_B_gt = np.zeros((len_B))
    for i, seg in enumerate(event_gt_B):
        start = event_gt_B[i][0]
        end = event_gt_B[i][1]
        event = event_gt_B[i][2]
        frame_events_B_gt[start:end] = event

    frame_events_B = np.zeros((len_B))
    for i, seg in enumerate(segs_B):
        event = event_B[i][0]
        start = seg[0]
        end = seg[1]
        frame_events_B[start:end] = event

    err_event = np.sum(frame_events_T != frame_events_T_gt) + np.sum(frame_events_B != frame_events_B_gt)

    with open(op.join(args.save_path, "RES_para_{}_clip_{}.p".format(para_idx, clip_idx)), 'wb') as f:
        pickle.dump([para, clip, Tree_best, err_seg, err_event], f)

def test_best_Tree(atomic_net, event_net, clip_idx, clip, args):

    pid = threading.get_ident()
    print('pid: {} clip {}'.format(pid, clip))

    beam_search = BeamSearch(args, event_net, atomic_net)

    beam_search.init(clip)
    while True:
        Tree_best=beam_search.tree_grow()
        if Tree_best is not None:
            break

    with open(op.join(args.save_path, clip), 'wb') as f:
        pickle.dump(Tree_best, f, protocol=2)

from select_para import select_para


def parse_arguments():

    parser=argparse.ArgumentParser(description='')
    # path
    home_path='/home/lfan/Dropbox/Projects/NIPS20/'
    home_path2='/media/lfan/HDD/NIPS20/'
    parser.add_argument('--project-path',default = home_path)
    parser.add_argument('--project-path2', default=home_path2)
    parser.add_argument('--data-path', default=home_path+'data/')
    parser.add_argument('--data-path2', default=home_path2 + 'data/')
    parser.add_argument('--img-path', default=home_path+'annotations/')
    parser.add_argument('--save-path', default='')
    parser.add_argument('--init-cps', default='/media/lfan/HDD/NIPS20/data/init_cps/CPS_CES_0908.p')
    parser.add_argument('--stat-path', default=home_path+'data/stat/')
    parser.add_argument('--attmat-path', default=home_path+'data/record_attention_matrix/')
    parser.add_argument('--cate-path', default=home_path2+'data/track_cate/')
    parser.add_argument('--tracker-bbox', default=home_path2+'data/tracker_record_bbox/')
    parser.add_argument('--battery-bbox', default=home_path2+'data/record_bbox/')
    parser.add_argument('--obj-bbox', default=home_path2+'data/post_neighbor_smooth_newseq/')
    parser.add_argument('--ednet-path', default=home_path+'model/ednet_tuned_best.pth')
    parser.add_argument('--atomic-path', default=home_path+'model/atomic_best.pth')
    parser.add_argument('--seg-label', default=home_path + 'data/segment_labels/')
    parser.add_argument('--feature-single', default=home_path2 + 'data/feature_single/')
    parser.add_argument('--save-event-score', default='/media/lfan/HDD/NIPS20/data/event_score_all_0906/')

    # parameter
    parser.add_argument('--lambda-1', default=1)
    parser.add_argument('--lambda-2', default=1)
    parser.add_argument('--lambda-3', default=1)
    parser.add_argument('--lambda-4', default=1)
    parser.add_argument('--lambda-5', default=1)
    parser.add_argument('--lambda-6', default=1)
    parser.add_argument('--lambda-7', default=1)
    parser.add_argument('--lambda-8', default=1)
    parser.add_argument('--lambda-9', default=1)
    parser.add_argument('--search-N-cp', default=5)
    parser.add_argument('--topN', default=3)

    parser.add_argument('--hist-bin', default=10)
    parser.add_argument('--seg-alpha', default=50)

    # others
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--ip', default='192.168.1.17')
    parser.add_argument('--port', default=1234)
    parser.add_argument('--resume',default=False)
    parser.add_argument('--test-func-batch-size', default=16)
    parser.add_argument('--gt-mode', default=True)

    return parser.parse_args()

if __name__ == "__main__":

    args=parse_arguments()
    args.save_path=op.join(args.project_path2, 'BestTree_ces')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Parallel(n_jobs=-1)(delayed(test_best_Tree)(atomic_event_net, event_net, clip_idx, clip, args) for clip_idx, clip in enumerate(clips_all))

    atomic_event_net=None
    event_net=None

    para_bank = [{'topN': 4, 'lambda_1': 1, 'lambda_2': 0.5, 'lambda_3': 1, 'lambda_5': 5, 'lambda_6': 5, 'lambda_7': 5,
     'lambda_8': 1, 'search_N_cp': 5}]

    with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    random.seed(0)
    random.shuffle(clips_with_gt_event)

    para_all=para_bank
    clips_all=clips_with_gt_event

    combinations=list(product(range(len(para_all)), range(len(clips_all))))
    print('There are {} tasks in total. {} clips, {} paras'.format(len(combinations), len(clips_all), len(para_all)))
    Parallel(n_jobs=1)(delayed(finetune_para)(atomic_event_net, event_net, comb[1], clips_all[comb[1]], comb[0], para_all[comb[0]], args, len(clips_all), len(para_all)) for _,  comb in enumerate(combinations))


    select_para(args)