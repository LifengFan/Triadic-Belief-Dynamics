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
import os, os.path
from metric import *
from BeamSearchClass import *

def generate_para_bank(args):
    #  build parameter bank
    para_bank=[]
    for topN in [5]:
        for lambda_2 in [0.1, 1, 10]:
            for lambda_3 in [0.1, 1, 10]:
                for lambda_4 in [0.1, 1, 10]:
                    for lambda_5 in [0.1, 1, 10]:
                        for lambda_6 in [0.1, 1, 10]:
                            for beta_1 in [0.1, 1, 10]:
                                for beta_2 in [0.1, 1, 10]:
                                    for beta_3 in [0.1, 1, 10]:
                                        for hist_bin in [10]:
                                            for search_N_cp in [5]:
                                                para_bank.append({'topN':topN, 'lambda_2':lambda_2, 'lambda_3':lambda_3,
                                                                  'lambda_4':lambda_4, 'lambda_5':lambda_5,'lambda_6':lambda_6,
                                                                  'beta_1':beta_1, 'beta_2':beta_2, 'beta_3':beta_3, 'hist_bin':hist_bin, 'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    return para_bank

def finetune_para(clip_list, para_bank, args):

    beam_search = BeamSearch(args)

    if args.resume:
        try:
            with open(op.join(args.save_path, 'resume_rec.p'), 'rb') as f:
                resume_rec=pickle.load(f)
                i_para_sp, i_clip_sp, ERR_PARA, cnt_clip, err_seg, err_event=resume_rec
        except:
            print("!"*10)
            print("\033[31m ERROR: no correct resume_rec file! \033[0m")
            i_para_sp = 0
            i_clip_sp = 0
            ERR_PARA=[]
            cnt_clip=0
            err_seg=0
            err_event=0
    else:
        i_para_sp=0
        i_clip_sp=0
        ERR_PARA=0
        cnt_clip = 0
        err_seg = 0
        err_event = 0

    for i_para in range(i_para_sp, len(para_bank)):
        para=para_bank[i_para]
        print("="*64)
        print("Parameter set: {}/{}".format(i_para, len(para_bank)))
        print("  ")
        for k, v in para.items():
            print("{} -- {}".format(k, v))
        print("=" * 64)

        args.topN=para['topN']
        args.lambda_2=para['lambda_2']
        args.lambda_3=para['lambda_3']
        args.lambda_4=para['lambda_4']
        args.lambda_5=para['lambda_5']
        args.lambda_6=para['lambda_6']
        args.beta_1=para['beta_1']
        args.beta_2=para['beta_2']
        args.beta_3=para['beta_3']
        args.hist_bin=para['hist_bin']
        args.search_N_cp=para['search_N_cp']

        finetune_save_path=op.join(args.save_path, 'para_{}'.format(i_para))
        if not os.path.exists(finetune_save_path):
            os.makedirs(finetune_save_path)

        if i_para>i_para_sp:
            i_clip_sp=0
        for i_clip in range(i_clip_sp, len(clip_list)):
            if i_clip==0:
                cnt_clip = 0
                err_seg = 0
                err_event = 0

            clip=clip_list[i_clip]

            with open(op.join(args.save_path, 'resume_rec.p'), 'wb') as f:
                pickle.dump([i_para, i_clip, ERR_PARA,cnt_clip, err_seg, err_event], f)

            beam_search.init(clip)

            #for i_video_sep, video_sep in enumerate(beam_search.video_seps):

            print(" Now - para set {}/{}, video {}/{} ({})".format(i_para, len(para_bank), i_clip, len(clip_list), clip))
            print("="*74)

            # beam search
            while True:
                Tree_best=beam_search.tree_grow()
                if Tree_best is not None:
                    break
            with open(op.join(finetune_save_path, 'Tree_best_'+clip), 'wb') as f:
                pickle.dump(Tree_best, f)

            # evaluation
            cps_T=Tree_best['T']['cp']
            cps_B=Tree_best['B']['cp']
            event_T=Tree_best['T']['event']
            event_B=Tree_best['B']['event']

            # seg
            segs_T=beam_search.cps2segs(cps_T)
            segs_B=beam_search.cps2segs(cps_B)
            tracker_gt_seg = event_seg_tracker[clip.split('.')[0]]
            battery_gt_seg = event_seg_battery[clip.split('.')[0]]
            curr_tracker_gt_seg = beam_search.find_sep_gt(tracker_gt_seg, segs_T)
            curr_battery_gt_seg = beam_search.find_sep_gt(battery_gt_seg, segs_B)

            err_seg += segment_error(segs_T, curr_tracker_gt_seg, args.seg_alpha) + segment_error(segs_B, curr_battery_gt_seg,args.seg_alpha)

            # event
            # tracker
            event_gt_T = event_seg_tracker[clip.split('.')[0]]
            len_T = event_gt_T[-1][1]
            frame_events_T_gt = np.zeros((1, len_T))
            for i, seg in enumerate(event_gt_T):
                start = event_gt_T[i][0]
                end = event_gt_T[i][1]
                event = event_gt_T[i][2]
                frame_events_T_gt[start:end] = event

            frame_events_T = np.zeros((1, len_T))
            for i, seg in enumerate(segs_T):
                event = event_T[i][0]
                start = seg[0]
                end = seg[1]
                frame_events_T[start:end] = event

            # battery
            event_gt_B = event_seg_battery[clip.split('.')[0]]
            len_B = event_gt_B[-1][1]
            frame_events_B_gt = np.zeros((1, len_B))
            for i, seg in enumerate(event_gt_B):
                start = event_gt_B[i][0]
                end = event_gt_B[i][1]
                event = event_gt_B[i][2]
                frame_events_B_gt[start:end] = event

            frame_events_B = np.zeros((1, len_B))
            for i, seg in enumerate(segs_B):
                event = event_B[i][0]
                start = seg[0]
                end = seg[1]
                frame_events_B[start:end] = event

            err_event += np.sum(frame_events_T != frame_events_T_gt) + np.sum(frame_events_B != frame_events_B_gt)
            cnt_clip += 1

        # current para
        ERR_PARA.append((err_seg + err_event)/float(cnt_clip))
        print("=" * 64)
        print("Finished parameter set {}, the err is {}".format(i_para, ERR_PARA[-1]))
        # top N para
        err_index = np.argsort(ERR_PARA)
        print("="*64)
        print("The Best Five Para Till Now:")
        print(np.array(para_bank)[err_index[:5]])
        print("Corresponding Err:")
        print(np.array(ERR_PARA)[err_index[:5]])

        with open(op.join(finetune_save_path, "para_err.p"), 'wb') as f:
            pickle.dump([i_para, para, ERR_PARA[-1]], f)
        with open(op.join(args.save_path, "ERR_PARA.p"), 'wb') as f: # to update this file constantly
            pickle.dump(ERR_PARA, f)

    with open(op.join(args.save_path, "ERR_PARA.p"), 'wb') as f:
        pickle.dump(ERR_PARA, f)

def parse_arguments():
    parser=argparse.ArgumentParser(description='')
    # path
    home_path='/home/lfan/Dropbox/Projects/NIPS20/'
    parser.add_argument('--project-path',default = home_path)
    parser.add_argument('--data-path', default=home_path+'data/')
    parser.add_argument('--img-path', default=home_path+'annotations/')
    parser.add_argument('--save-path', default=home_path+'result/BeamSearch/')
    parser.add_argument('--init-cps', default=home_path+'data/cps_comb1.p')
    parser.add_argument('--stat-path', default=home_path+'data/stat/')
    parser.add_argument('--attmat-path', default=home_path+'data/record_attention_matrix/')
    parser.add_argument('--cate-path', default=home_path+'data/track_cate/')
    parser.add_argument('--tracker-bbox', default=home_path+'data/3d_pose2gaze/tracker_record_bbox/')
    parser.add_argument('--battery-bbox', default=home_path+'data/3d_pose2gaze/record_bbox/')
    parser.add_argument('--obj-bbox', default=home_path+'post_neighbor_smooth_newseq/')
    parser.add_argument('--event-model-path', default=home_path+'code/model_event.pth')
    parser.add_argument('--atomic-event-path', default=home_path+'code/model_best_atomic.pth')
    parser.add_argument('--resume',default=True) # to resume from the last point
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
    parser.add_argument('--search-N-cp', default=5)
    parser.add_argument('--topN', default=5)

    parser.add_argument('--hist-bin', default=10)
    parser.add_argument('--seg-alpha', default=10)
    # others
    parser.add_argument('--cuda', default=True)

    return parser.parse_args()

def run():

    print("="*74)
    print('/'*5)
    print("/ \033[31m [Important!] Don't forget to change your project name accordingly! \033[0m ")
    print('/' * 5)
    print("=" * 74)

    args = parse_arguments()

    project_name='finetune_para_lfan_05141800'
    args.save_path=args.save_path+project_name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    clip_list=clips_with_gt_event[:3]

    para_bank=generate_para_bank(args)
    para_bank=para_bank[:1000]

    with open(op.join(args.save_path, 'experiment_setting.p'), 'wb') as f:
        pickle.dump([clip_list, para_bank, args],f)

    finetune_para(clip_list, para_bank, args)

if __name__ == "__main__":

    run()














