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


class EventScore(object):

    def __init__(self, atomic_net, event_net, args):

        self.init_cps_all=pickle.load(open(args.init_cps, 'rb'), encoding='latin1')
        self.args = args
        self.event_net=event_net
        self.atomic_net=atomic_net

    def run(self, clip):

        self.clip = clip
        self.clip_len = clips_len[self.clip]

        self.init_cps_T=self.init_cps_all[0][self.clip]
        self.init_cps_T.append(self.clip_len)
        self.init_cps_T=list(np.unique(self.init_cps_T))

        self.init_cps_B = self.init_cps_all[1][self.clip]
        self.init_cps_B.append(self.clip_len)
        self.init_cps_B = list(np.unique(self.init_cps_B))

        self.find_segs()

        with open(self.args.tracker_bbox + clip, 'rb') as f:
            self.person_tracker_bbox = pickle.load(f, encoding='latin1')
        with open(self.args.battery_bbox + clip, 'rb') as f:
            self.person_battery_bbox = pickle.load(f, encoding='latin1')

        # attmat
        with open(self.args.attmat_path + clip, 'rb') as f:
            self.attmat_obj = pickle.load(f, encoding='latin1')
        with open(self.args.cate_path + clip, 'rb') as f:
            self.category = pickle.load(f, encoding='latin1')

        if not op.exists(op.join(args.save_event_score, clip.split('.')[0])):
            os.makedirs(op.join(args.save_event_score, clip.split('.')[0]))

        if not op.exists(op.join(args.save_event_score, clip.split('.')[0], 'tracker')):
            os.makedirs(op.join(args.save_event_score, clip.split('.')[0], 'tracker'))

        if not op.exists(op.join(args.save_event_score, clip.split('.')[0], 'battery')):
            os.makedirs(op.join(args.save_event_score, clip.split('.')[0], 'battery'))

        for seg in self.segs_T:

            if os.path.exists(op.join(args.save_event_score, self.clip.split('.')[0], 'tracker',
                                      '{}_{}.p'.format(seg[0], seg[1]))):
                continue

            outputs = self.event_score(seg[0], seg[1])

            with open(op.join(args.save_event_score, self.clip.split('.')[0], 'tracker',
                              '{}_{}.p'.format(seg[0], seg[1])), 'wb') as f:
                pickle.dump(outputs, f)

            print('[]/[] clip{} T seg {}'.format(self.clip, seg))

        for seg in self.segs_B:

            if os.path.exists(op.join(args.save_event_score, self.clip.split('.')[0], 'tracker',
                                      '{}_{}.p'.format(seg[0], seg[1]))):
                continue

            outputs = self.event_score(seg[0], seg[1])

            with open(op.join(args.save_event_score, self.clip.split('.')[0], 'battery',
                              '{}_{}.p'.format(seg[0], seg[1])), 'wb') as f:
                pickle.dump(outputs, f)
            print('[]/[] clip {} B seg {}'.format(self.clip, seg))

    def find_segs(self):

        self.segs_T=[]
        for id, start in enumerate(self.init_cps_T):
            for id2 in range(id+1, min(id+self.args.search_N_cp, len(self.init_cps_T))):
                end=self.init_cps_T[id2]
                self.segs_T.append([start,end])

        self.segs_B=[]
        for id, start in enumerate(self.init_cps_B):
            for id2 in range(id+1, min(id+self.args.search_N_cp, len(self.init_cps_B))):
                end=self.init_cps_B[id2]
                self.segs_B.append([start,end])

    def event_score(self, start, end):

        test_seq = find_test_seq_over(self.attmat_obj, start, end, self.category, self.clip,
                                      self.person_tracker_bbox, self.person_battery_bbox)

        if len(test_seq) == 0:
            return [[0.33, 0.33, 0.33]]
        else:
            test_set = mydataset_atomic_with_label_first_view_reformat(test_seq, self.args)
            test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic_first_view,batch_size=args.test_func_batch_size, shuffle=False)
            test_results = test(test_loader, self.atomic_net, self.args)
            input1, input2 = merge_results(test_results)
            input1_pad = np.zeros((1, 50))
            input2_pad = np.zeros((1, 50))
            for i in range(len(input1)):
                input1_pad[0, i] = input1[i]
            for i in range(len(input2)):
                input2_pad[0, i] = input2[i]
            if args.cuda:
                input1s = torch.tensor(input1_pad).float().cuda()
                input2s = torch.tensor(input2_pad).float().cuda()
            else:
                input1s = torch.tensor(input1_pad).float()
                input2s = torch.tensor(input2_pad).float()
            with torch.no_grad():
                outputs = self.event_net(input1s, input2s)
                outputs = torch.softmax(outputs,dim=-1)
            outputs = outputs.data.cpu().numpy()
            return outputs


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
    parser.add_argument('--save-root', default='/media/lfan/HDD/NIPS20/Result/event_score_cps_new_0601/')
    parser.add_argument('--save-path', default='/media/lfan/HDD/NIPS20/Result/event_score_cps_new_0601/')
    parser.add_argument('--init-cps', default='/media/lfan/HDD/NIPS20/data/init_cps/CPS_NEW_0601.p')
    parser.add_argument('--stat-path', default=home_path+'data/stat/')
    parser.add_argument('--attmat-path', default=home_path+'data/record_attention_matrix/')
    parser.add_argument('--cate-path', default=home_path2+'data/track_cate/')
    parser.add_argument('--tracker-bbox', default=home_path2+'data/tracker_record_bbox/')
    parser.add_argument('--battery-bbox', default=home_path2+'data/record_bbox/')
    parser.add_argument('--obj-bbox', default=home_path2+'data/post_neighbor_smooth_newseq/')
    parser.add_argument('--ednet-path', default=home_path+'code/ednet_tuned_best.pth')
    parser.add_argument('--atomic-path', default=home_path+'code/atomic_best.pth')
    parser.add_argument('--seg-label', default=home_path + 'data/segment_labels/')
    parser.add_argument('--feature-single', default=home_path2 + 'data/feature_single/')
    parser.add_argument('--save-event-score', default='/media/lfan/HDD/NIPS20/data/event_score_all/')

    # parameter
    parser.add_argument('--lambda-1', default=1)
    parser.add_argument('--lambda-2', default=1)
    parser.add_argument('--lambda-3', default=1)
    parser.add_argument('--lambda-4', default=1)
    parser.add_argument('--lambda-5', default=1)
    parser.add_argument('--lambda-6', default=1)
    parser.add_argument('--beta-1', default=1)
    parser.add_argument('--beta-2', default=1)
    parser.add_argument('--beta-3', default=1)
    parser.add_argument('--search-N-cp', default=6)
    parser.add_argument('--topN', default=1)

    parser.add_argument('--hist-bin', default=10)
    parser.add_argument('--seg-alpha', default=10) #todo: check the alpha here!

    # others
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--ip', default='192.168.1.17')
    parser.add_argument('--port', default=1234)
    parser.add_argument('--resume',default=False) # to resume from the last point
    parser.add_argument('--test-func-batch-size', default=16)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    atomic_net = Atomic_node_only_lstm_first_view()
    load_best_checkpoint(atomic_net, path=args.atomic_path)
    if args.cuda and torch.cuda.is_available():
        atomic_net.cuda()
    atomic_net.eval()

    event_net=EDNet()
    event_net.load_state_dict(torch.load(args.ednet_path))
    if args.cuda and torch.cuda.is_available():
        event_net.cuda()
    event_net.eval()

    event_score = EventScore(atomic_net, event_net, args)

    Parallel(n_jobs=1)(delayed(event_score.run)(clip) for _, clip in enumerate(mind_test_clips))

