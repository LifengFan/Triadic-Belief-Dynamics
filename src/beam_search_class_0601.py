from metadata import *
import pywt
from overall_event_get_input import *
import copy
import os, os.path
from metric import *
from itertools import product

class BeamSearch_P(object):

    # ================
    # init functions
    # ================
    def __init__(self, args):
        self.args = args
        self.epsilon = 0.000001
        self.init_cps_all = pickle.load(open(args.init_cps, 'rb'), encoding='latin1')
        self.event_trans_table = pickle.load(open(os.path.join(args.stat_path, 'event_trans_normalized.p'), 'rb'), encoding='latin1')

    def init(self, clip):

        self.clip = clip
        self.clip_length = clips_len[clip]

        # initial cps
        self.init_cps_T = self.init_cps_all[0][self.clip]
        self.init_cps_B = self.init_cps_all[1][self.clip]
        # add the final frame
        self.init_cps_T.append(self.clip_length)
        self.init_cps_T=list(np.unique(self.init_cps_T))
        self.init_cps_B.append(self.clip_length)
        self.init_cps_B = list(np.unique(self.init_cps_B))

        with open(self.args.tracker_bbox + clip, 'rb') as f:
            self.person_tracker_bbox = pickle.load(f, encoding='latin1')
        with open(self.args.battery_bbox + clip, 'rb') as f:
            self.person_battery_bbox = pickle.load(f, encoding='latin1')

        # attmat
        with open(self.args.attmat_path + clip, 'rb') as f:
            self.attmat_obj = pickle.load(f, encoding='latin1')

        with open(self.args.cate_path + clip, 'rb') as f:
            self.category = pickle.load(f, encoding='latin1')

        # feature
        with open(os.path.join(self.args.data_path2, 'feature_single', clip), 'rb') as f:
            self.feature_single = pickle.load(f, encoding='latin1')

        if tracker_skeID[clip.split('.')[0]] == 'skele1.p':
            self.T_skeID = 1
            self.B_skeID = 2
        elif tracker_skeID[clip.split('.')[0]] == 'skele2.p':
            self.T_skeID = 2
            self.B_skeID = 1

        # init root node and tree for each new clip
        self.root_node = {'T': {'cp': [0], 'event': [], 'mind': [], 'event_vec':[]}, 'B': {'cp': [0], 'event': [], 'mind': [], 'event_vec':[]},
                          'score':{'prior_aggr':{'cnt_joint':0., 'cnt_single':0.},
                                   'prior_event':{'e_T':0., 'cnt_T':0., 'e_S':0., 'cnt_S':0., 'T_last_type':None, 'B_last_type':None},
                                   'prior_fluent':0.,
                                   'like_P':{'T_D_i':0., 'T_CNT_i':0., 'B_D_i':0., 'B_CNT_i':0., 'T_D_t':0., 'T_CNT_t':0., 'B_D_t':0., 'B_CNT_t':0., 'last_T_hist':None, 'last_B_hist':None, 'D_s':0., 'CNT_s':0.},
                                   'like_E':{'e_E':0., 'CNT':0.},
                                   'like_M':0,
                                   'T_update_flag':0,
                                   'B_update_flag':0
                                   }
                          } #flag indicates whether it's updated.
        self.Tree = {'nodes': [self.root_node], 'level': 0}

    # ================
    # utils functions
    # ================
    def seg2frame(self, node):
        segs = node['cp']
        events = node['event']
        #assert (len(segs) - 1) == len(events)

        frame_labels = np.zeros((segs[-1]))
        for seg_id, seg in enumerate(segs[:-1]):
            event = events[seg_id][0]
            start = seg
            end = segs[seg_id + 1]
            frame_labels[start: end] = event

        return frame_labels

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

        node['score']['like_P']['T_D_i']= T_D_i
        node['score']['like_P']['T_CNT_i']= T_CNT_i

        # inter particle distance - T
        T_D_t = node['score']['like_P']['T_D_t']
        T_CNT_t = node['score']['like_P']['T_CNT_t']
        last_T_hist= node['score']['like_P']['last_T_hist']
        T_hist = self.temporal2freq(feature)
        if last_T_hist is not None:
            T_D_t += np.linalg.norm(last_T_hist - T_hist)/312.
            T_CNT_t += 1

        node['score']['like_P']['T_D_t']=T_D_t
        node['score']['like_P']['T_CNT_t']=T_CNT_t
        node['score']['like_P']['last_T_hist']=T_hist

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

        node['score']['like_P']['B_D_i']= B_D_i
        node['score']['like_P']['B_CNT_i']= B_CNT_i

        # inter particle distance - T
        B_D_t = node['score']['like_P']['B_D_t']
        B_CNT_t = node['score']['like_P']['B_CNT_t']
        last_B_hist= node['score']['like_P']['last_B_hist']
        B_hist = self.temporal2freq(feature)
        if last_B_hist is not None:
            B_D_t += np.linalg.norm(last_B_hist - B_hist)/312.
            B_CNT_t += 1

        node['score']['like_P']['B_D_t']=B_D_t
        node['score']['like_P']['B_CNT_t']=B_CNT_t
        node['score']['like_P']['last_B_hist']=B_hist

        e_P = self.args.beta_1 * (T_D_i / (T_CNT_i + self.epsilon) + B_D_i / (B_CNT_i + self.epsilon)) - self.args.beta_3 * (T_D_t / (T_CNT_t + self.epsilon) + B_D_t / (B_CNT_t + self.epsilon))

        return e_P, node

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
        all_possible_nodes_new=[]
        for idx, node in enumerate(all_possible_path):
            e_aggr, node = self.prior_energy_aggr(node)
            e_P, node = self.likelihood_energy_P(node)

            node_score = -e_aggr - e_P
            score_all.append(node_score)
            node_ids.append(idx)
            all_possible_nodes_new.append(node)

        ordered_index = list(np.argsort(np.array(score_all))[::-1])
        selected_index = ordered_index[:self.args.topN]
        node_ids = np.array(node_ids)
        top_node_ids = node_ids[selected_index]

        # assert len(top_node_ids)>0, 'no valid top nodes!'
        if len(top_node_ids)>0:
            self.Tree['nodes'] = []
            for node_id in top_node_ids:
                node = all_possible_nodes_new[node_id]
                self.Tree['nodes'].append(node)
            return True
        else:
            self.check_event_flag=False
            print('err! no valid top nodes! first time')
            score_all = []
            node_ids = []
            all_possible_nodes_new = []
            for idx, node in enumerate(all_possible_path):
                # calculate score for the current node/path
                e_aggr, node = self.prior_energy_aggr(node)
                e_P, node = self.likelihood_energy_P(node)
                node_score = -e_aggr - e_P
                score_all.append(node_score)
                node_ids.append(idx)

                all_possible_nodes_new.append(node)

            ordered_index = list(np.argsort(np.array(score_all))[::-1])
            selected_index = ordered_index[:self.args.topN]
            node_ids = np.array(node_ids)
            top_node_ids = node_ids[selected_index]

            if len(top_node_ids)>0:
                self.Tree['nodes'] = []
                for node_id in top_node_ids:
                    node = all_possible_nodes_new[node_id]
                    self.Tree['nodes'].append(node)
                return True
            else:
                print('err! no valid top nodes! second time')
                return False

    def node_expand(self, parent_node):
        '''
        input:
            parent checkpoint node
        :return:
            all possible children nodes
        '''
        t_node_id = parent_node['T']['cp'][-1]
        possible_t_cp =[]
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
        #print("I am process {}-- into tree grow".format(process_i))
        '''
        input:
            current top N possible path
        :return:
            new top N possible path
        '''
        all_possible_nodes = []
        for idx, parent_node in enumerate(self.Tree['nodes']):
            # find possible child nodes of the current node
            possible_t_cp, possible_b_cp = self.node_expand(parent_node)
            search_N_cp_T = min(len(possible_t_cp), self.args.search_N_cp)
            search_N_cp_B = min(len(possible_b_cp), self.args.search_N_cp)

            # all possible paths
            if len(possible_t_cp) >=1 and len(possible_b_cp) >=1:
                for cp_t_id in possible_t_cp[:search_N_cp_T]:
                    for cp_b_id in possible_b_cp[:search_N_cp_B]:

                            new_node = copy.deepcopy(parent_node)
                            new_node['T']['cp'].append(cp_t_id)
                            new_node['B']['cp'].append(cp_b_id)
                            new_node['score']['T_update_flag'] = 1
                            new_node['score']['B_update_flag'] = 1
                            all_possible_nodes.append(new_node)

            elif len(possible_t_cp) == 0 and len(possible_b_cp) >=1:
                for cp_b_id in possible_b_cp[:search_N_cp_B]:

                    new_node1, new_node2, new_node3 = copy.deepcopy(parent_node), copy.deepcopy(
                        parent_node), copy.deepcopy(parent_node)

                    # add new cp
                    new_node1['B']['cp'].append(cp_b_id)
                    new_node2['B']['cp'].append(cp_b_id)
                    new_node3['B']['cp'].append(cp_b_id)
                    new_node1['score']['T_update_flag'] = 0
                    new_node1['score']['B_update_flag'] = 1
                    new_node2['score']['T_update_flag'] = 0
                    new_node2['score']['B_update_flag'] = 1
                    new_node3['score']['T_update_flag'] = 0
                    new_node3['score']['B_update_flag'] = 1
                    all_possible_nodes.append(new_node1)
                    all_possible_nodes.append(new_node2)
                    all_possible_nodes.append(new_node3)

            elif len(possible_t_cp) >= 1 and len(possible_b_cp) == 0:
                for cp_t_id in possible_t_cp[:search_N_cp_T]:

                    new_node1, new_node2, new_node3 = copy.deepcopy(parent_node), copy.deepcopy(
                        parent_node), copy.deepcopy(parent_node)

                    # add new cp
                    new_node1['T']['cp'].append(cp_t_id)
                    new_node2['T']['cp'].append(cp_t_id)
                    new_node3['T']['cp'].append(cp_t_id)
                    new_node1['score']['T_update_flag'] = 1
                    new_node1['score']['B_update_flag'] = 0
                    new_node2['score']['T_update_flag'] = 1
                    new_node2['score']['B_update_flag'] = 0
                    new_node3['score']['T_update_flag'] = 1
                    new_node3['score']['B_update_flag'] = 0
                    all_possible_nodes.append(new_node1)
                    all_possible_nodes.append(new_node2)
                    all_possible_nodes.append(new_node3)

        if len(all_possible_nodes) == 0:
            return self.Tree['nodes'][0]
        else:
            flag=self.tree_prune(all_possible_nodes)
            self.Tree['level'] += 1
            if flag==True:
                return None
            elif flag==False:
                return self.Tree['nodes'][0]

class BeamSearch_E(object):

    # ================
    # init functions
    # ================
    def __init__(self, args, event_net, atomic_net):

        self.args = args
        self.epsilon = 0.000001
        self.init_cps_all = pickle.load(open(args.init_cps, 'rb'), encoding='latin1')

        self.event_net=event_net
        self.atomic_net=atomic_net

        self.event_trans_table = pickle.load(open(os.path.join(args.stat_path, 'event_trans_normalized.p'), 'rb'), encoding='latin1')

    def init(self, clip, Tree_best_P):

        self.clip = clip
        self.clip_length = clips_len[clip]
        self.Tree_best_P=Tree_best_P

        if tracker_skeID[clip.split('.')[0]] == 'skele1.p':
            self.T_skeID = 1
            self.B_skeID = 2
        elif tracker_skeID[clip.split('.')[0]] == 'skele2.p':
            self.T_skeID = 2
            self.B_skeID = 1

        # initial cps
        self.init_cps_T = self.Tree_best_P['T']['cp']
        self.init_cps_B = self.Tree_best_P['B']['cp']
        # add the final frame
        self.init_cps_T.append(self.clip_length)
        self.init_cps_T=list(np.unique(self.init_cps_T))
        self.init_cps_B.append(self.clip_length)
        self.init_cps_B = list(np.unique(self.init_cps_B))

        with open(self.args.tracker_bbox + clip, 'rb') as f:
            self.person_tracker_bbox = pickle.load(f, encoding='latin1')
        with open(self.args.battery_bbox + clip, 'rb') as f:
            self.person_battery_bbox = pickle.load(f, encoding='latin1')

        # attmat
        with open(self.args.attmat_path + clip, 'rb') as f:
            self.attmat_obj = pickle.load(f, encoding='latin1')

        with open(self.args.cate_path + clip, 'rb') as f:
            self.category = pickle.load(f, encoding='latin1')

        # feature
        with open(os.path.join(self.args.data_path2, 'feature_single', clip), 'rb') as f:
            self.feature_single = pickle.load(f, encoding='latin1')

        # init root node and tree for each new clip
        self.root_node = {'T': {'cp': [0], 'event': [], 'mind': [], 'event_vec':[]}, 'B': {'cp': [0], 'event': [], 'mind': [], 'event_vec':[]},
                          'score':{'prior_aggr':{'cnt_joint':0., 'cnt_single':0.},
                                   'prior_event':{'e_T':0., 'cnt_T':0., 'e_S':0., 'cnt_S':0., 'T_last_type':None, 'B_last_type':None},
                                   'prior_fluent':0.,
                                   'like_P':{'T_D_i':0., 'T_CNT_i':0., 'B_D_i':0., 'B_CNT_i':0., 'T_D_t':0., 'T_CNT_t':0., 'B_D_t':0., 'B_CNT_t':0., 'last_T_hist':None, 'last_B_hist':None, 'D_s':0., 'CNT_s':0.},
                                   'like_E':{'e_E':0., 'CNT':0.},
                                   'like_M':0,
                                   'T_update_flag':0,
                                   'B_update_flag':0
                                   }
                          } # flag indicates whether it's updated.
        self.Tree = {'nodes': [self.root_node], 'level': 0}

        self.check_event_flag=True

    # ================
    # utils functions
    # ================
    def seg2frame(self, node):
        segs = node['cp']
        events = node['event']
        #assert (len(segs) - 1) == len(events)

        frame_labels = np.zeros((segs[-1]))
        for seg_id, seg in enumerate(segs[:-1]):
            event = events[seg_id][0]
            start = seg
            end = segs[seg_id + 1]
            frame_labels[start: end] = event

        return frame_labels

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

    def prior_energy_event(self, node):
        e_T = node['score']['prior_event']['e_T']
        cnt_T = node['score']['prior_event']['cnt_T']
        T_last_type=node['score']['prior_event']['T_last_type']
        B_last_type=node['score']['prior_event']['B_last_type']

        # temporal transition
        T_new_type = node['T']['event'][-1][0]
        if T_last_type is not None:
            trans_key = (T_last_type, T_new_type)
            e_T += self.event_trans_table[trans_key]
            cnt_T += 1
        T_last_type=T_new_type

        B_new_type = node['B']['event'][-1][0]
        if B_last_type is not None:
            trans_key = (B_last_type, B_new_type)
            e_T += self.event_trans_table[trans_key]
            cnt_T += 1
        B_last_type=B_new_type

        node['score']['prior_event']['e_T']=e_T
        node['score']['prior_event']['cnt_T']=cnt_T
        node['score']['prior_event']['T_last_type']=T_last_type
        node['score']['prior_event']['B_last_type']=B_last_type

        e_T = e_T / (cnt_T + self.epsilon)

        # spatial concurrency
        e_S = node['score']['prior_event']['e_S']
        cnt_S = node['score']['prior_event']['cnt_S']

        segs_T = self.cps2segs(node['T']['cp'])
        segs_B = self.cps2segs(node['B']['cp'])

        if node['score']['T_update_flag']==1:
            for idx2 in range(len(segs_B)):
                if segs_T[-1][0] >= segs_B[idx2][1]:
                    continue
                elif segs_T[-1][1] <= segs_B[idx2][0]:
                    break
                else:
                    event_T = node['T']['event'][-1][0]
                    event_B = node['B']['event'][idx2][0]
                    if event_T == event_B:
                        e_S += 1
                        cnt_S += 1
                    else:
                        e_S += 0
                        cnt_S += 1

        if node['score']['B_update_flag']==1:
            for idx1 in range(len(segs_T)):
                if segs_T[idx1][0] >= segs_B[-1][1]:
                    break
                elif segs_T[idx1][1] <= segs_B[-1][0]:
                    continue
                else:
                    event_T = node['T']['event'][idx1][0]
                    event_B = node['B']['event'][-1][0]
                    if event_T == event_B:
                        e_S += 1
                        cnt_S += 1
                    else:
                        e_S += 0
                        cnt_S += 1
        if node['score']['T_update_flag']==1 and node['score']['B_update_flag']==1:
            event_T = node['T']['event'][-1][0]
            event_B = node['B']['event'][-1][0]
            if event_T == event_B:
                e_S -= 1
                cnt_S -= 1
            else:
                e_S -= 0
                cnt_S-= 1

        node['score']['prior_event']['e_S']=e_S
        node['score']['prior_event']['cnt_S']=cnt_S

        e_S = e_S / (cnt_S + self.epsilon)

        e_event = -self.args.lambda_4 * (e_T + self.epsilon) - self.args.lambda_5 * (e_S + self.epsilon)

        return e_event, node

    def likelihood_energy_E(self, node):

        e_E = node['score']['like_E']['e_E']
        CNT = node['score']['like_E']['CNT']

        if node['score']['T_update_flag'] == 1:
            e_E += node['T']['event'][-1][1]
            CNT += 1
        if node['score']['B_update_flag'] == 1:
            e_E += node['B']['event'][-1][1]
            CNT += 1

        node['score']['like_E']['e_E'] = e_E
        node['score']['like_E']['CNT'] = CNT

        e_E = e_E / CNT

        energy_E = -self.args.gamma_1 * e_E

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
        all_possible_nodes_new=[]

        for idx, node in enumerate(all_possible_path):
            e_event, node = self.prior_energy_event(node)
            e_E, node = self.likelihood_energy_E(node)
            node_score = - e_event - e_E
            score_all.append(node_score)
            node_ids.append(idx)
            all_possible_nodes_new.append(node)

        ordered_index = list(np.argsort(np.array(score_all))[::-1])
        selected_index = ordered_index[:self.args.topN]
        node_ids = np.array(node_ids)
        top_node_ids = node_ids[selected_index]

        if len(top_node_ids)>0:
            self.Tree['nodes'] = []
            for node_id in top_node_ids:
                node = all_possible_nodes_new[node_id]
                self.Tree['nodes'].append(node)
            return True
        else:
            self.check_event_flag=False
            print('err! no valid top nodes! first time')
            score_all = []
            node_ids = []
            all_possible_nodes_new = []
            for idx, node in enumerate(all_possible_path):
                e_event, node = self.prior_energy_event(node)
                e_E, node = self.likelihood_energy_E(node)
                node_score =  - e_event  - e_E
                score_all.append(node_score)
                node_ids.append(idx)

                all_possible_nodes_new.append(node)

            ordered_index = list(np.argsort(np.array(score_all))[::-1])
            selected_index = ordered_index[:self.args.topN]
            node_ids = np.array(node_ids)
            top_node_ids = node_ids[selected_index]

            if len(top_node_ids)>0:
                self.Tree['nodes'] = []
                for node_id in top_node_ids:
                    node = all_possible_nodes_new[node_id]
                    self.Tree['nodes'].append(node)
                return True
            else:
                print('err! no valid top nodes! second time')
                return False

    def node_expand(self, parent_node):
        '''
        input:
            parent checkpoint node
        :return:
            all possible children nodes
        '''
        t_node_id = parent_node['T']['cp'][-1]
        possible_t_cp =[]
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
        #print("I am process {}-- into tree grow".format(process_i))
        '''
        input:
            current top N possible path
        :return:
            new top N possible path
        '''
        all_possible_nodes = []
        for idx, parent_node in enumerate(self.Tree['nodes']):
            # find possible child nodes of the current node
            possible_t_cp, possible_b_cp = self.node_expand(parent_node)
            search_N_cp_T = 1
            search_N_cp_B = 1

            # all possible paths
            if len(possible_t_cp) >=1 and len(possible_b_cp) >=1:
                for cp_t_id in possible_t_cp[:search_N_cp_T]:
                    for cp_b_id in possible_b_cp[:search_N_cp_B]:

                        combinations = list(product([0, 1, 2], repeat=2))
                        for combination in combinations:
                            new_node = copy.deepcopy(parent_node)
                            new_node['T']['cp'].append(cp_t_id)
                            new_node['B']['cp'].append(cp_b_id)

                            start = parent_node['T']['cp'][-1]
                            end = cp_t_id
                            with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'tracker',
                                              '{}_{}.p'.format(start, end)), 'rb') as f:
                                outputs = pickle.load(f)
                            new_node['T']['event'].append([combination[0], outputs[0][combination[0]]])
                            new_node['T']['event_vec'].append(outputs[0])

                            start = parent_node['B']['cp'][-1]
                            end = cp_b_id
                            with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'battery',
                                              '{}_{}.p'.format(start, end)), 'rb') as f:
                                outputs = pickle.load(f)
                            new_node['B']['event'].append([combination[1], outputs[0][combination[1]]])
                            new_node['B']['event_vec'].append(outputs[0])

                            new_node['score']['T_update_flag'] = 1
                            new_node['score']['B_update_flag'] = 1

                            all_possible_nodes.append(new_node)

            elif len(possible_t_cp) == 0 and len(possible_b_cp) >=1:
                for cp_b_id in possible_b_cp[:search_N_cp_B]:

                    new_node1, new_node2, new_node3 = copy.deepcopy(parent_node), copy.deepcopy(
                        parent_node), copy.deepcopy(parent_node)

                    # add new cp
                    new_node1['B']['cp'].append(cp_b_id)
                    new_node2['B']['cp'].append(cp_b_id)
                    new_node3['B']['cp'].append(cp_b_id)

                    # predict event for current seg
                    # battery
                    start = parent_node['B']['cp'][-1]
                    end = cp_b_id
                    with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'battery',
                                      '{}_{}.p'.format(start, end)), 'rb') as f:
                        outputs = pickle.load(f)
                    new_node1['B']['event'].append([0, outputs[0][0]])
                    new_node1['B']['event_vec'].append(outputs[0])
                    new_node2['B']['event'].append([1, outputs[0][1]])
                    new_node2['B']['event_vec'].append(outputs[0])
                    new_node3['B']['event'].append([2, outputs[0][2]])
                    new_node3['B']['event_vec'].append(outputs[0])

                    new_node1['score']['T_update_flag'] = 0
                    new_node1['score']['B_update_flag'] = 1
                    new_node2['score']['T_update_flag'] = 0
                    new_node2['score']['B_update_flag'] = 1
                    new_node3['score']['T_update_flag'] = 0
                    new_node3['score']['B_update_flag'] = 1
                    all_possible_nodes.append(new_node1)
                    all_possible_nodes.append(new_node2)
                    all_possible_nodes.append(new_node3)
            elif len(possible_t_cp) >= 1 and len(possible_b_cp) == 0:

                for cp_t_id in possible_t_cp[:search_N_cp_T]:

                    new_node1, new_node2, new_node3 = copy.deepcopy(parent_node), copy.deepcopy(
                        parent_node), copy.deepcopy(parent_node)

                    # add new cp
                    new_node1['T']['cp'].append(cp_t_id)
                    new_node2['T']['cp'].append(cp_t_id)
                    new_node3['T']['cp'].append(cp_t_id)

                    # predict event for current seg
                    # battery
                    start = parent_node['T']['cp'][-1]
                    end = cp_t_id

                    with open(op.join(self.args.save_event_score, self.clip.split('.')[0], 'tracker',
                                      '{}_{}.p'.format(start, end)), 'rb') as f:
                        outputs = pickle.load(f)
                    new_node1['T']['event'].append([0, outputs[0][0]])
                    new_node1['T']['event_vec'].append(outputs[0])
                    new_node2['T']['event'].append([1, outputs[0][1]])
                    new_node2['T']['event_vec'].append(outputs[0])
                    new_node3['T']['event'].append([2, outputs[0][2]])
                    new_node3['T']['event_vec'].append(outputs[0])

                    new_node1['score']['T_update_flag'] = 1
                    new_node1['score']['B_update_flag'] = 0
                    new_node2['score']['T_update_flag'] = 1
                    new_node2['score']['B_update_flag'] = 0
                    new_node3['score']['T_update_flag'] = 1
                    new_node3['score']['B_update_flag'] = 0
                    all_possible_nodes.append(new_node1)
                    all_possible_nodes.append(new_node2)
                    all_possible_nodes.append(new_node3)

            else:
                all_possible_nodes.append(parent_node)

        if len(all_possible_nodes) == len(self.Tree['nodes']):
            #assert all_possible_nodes==self.Tree['nodes']
            #print("Beam Search comes to the end for clip {}. Total level is {}".format(self.clip, self.Tree['level']))
            return self.Tree['nodes'][0]
        else:
            flag=self.tree_prune(all_possible_nodes)
            self.Tree['level'] += 1
            if flag==True:
                return None
            elif flag==False:
                return self.Tree['nodes'][0]

