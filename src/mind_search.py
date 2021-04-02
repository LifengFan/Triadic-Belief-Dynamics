from metadata import *
import argparse
import pickle
import time
from threading import Thread, current_thread
import os, os.path
from joblib import Parallel, delayed
from mind_model import *
import pandas as pd
from itertools import product
import os.path as op
from mind_model import *
import random
from utils import *
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

class BeamSearch_Mind(object):

    # ================
    # init functions
    # ================
    def __init__(self, args, mind_net):
        self.args = args
        self.epsilon = 0.000001
        self.mind_net = mind_net

    def init(self, clip):

        self.clip = clip  # with .p
        self.clip_length = clips_len[clip]

        with open(op.join(self.args.data_path, 'event_mind_prob.p'), 'rb') as f:
            self.event_mind_prob = pickle.load(f)

        for i in range(1, len(self.event_mind_prob)):
            count = []
            for key in self.event_mind_prob[i]:
                mind_count = np.array([self.event_mind_prob[i][key][0], self.event_mind_prob[i][key][1], self.event_mind_prob[i][key][2], self.event_mind_prob[i][key][3]])
                mind_count = mind_count/float(np.sum(mind_count))
                assert abs(np.sum(mind_count) - 1) < 0.001
                self.event_mind_prob[i][key] = mind_count

        with open(op.join(self.args.data_path, 'between_mind_prob.p'), 'rb') as f:
            self.between_mind_prob = pickle.load(f)

        count = 0
        for key in self.between_mind_prob.keys():
            count += self.between_mind_prob[key]

        count_list = np.empty(0)
        for key in self.between_mind_prob.keys():
            self.between_mind_prob[key] = self.between_mind_prob[key]/float(count)
            count_list = np.append(count_list, self.between_mind_prob[key])

        assert abs(1- np.sum(count_list)) < 0.001

        self.Tree = {'nodes': [], 'level': 0}

    # ================
    # tree functions
    # ================
    def tree_prune(self, new_node):
        '''
        input:
            all possible paths in this level
        :return:
            top N paths
        '''
        mind_sep = {}
        for key in new_node:
            if key == 'event':
                continue
            if key == 'valid_pool':
                continue
            if len(self.Tree['nodes']) == 0:
                valid_pool = np.array([1/2., 0, 0, 1/2.])
            else:
                valid_pool = self.Tree['nodes'][-1]['valid_pool'][key]

            likelihood = new_node[key]

            event_id = np.argmax(new_node['event'])
            event_prior = self.event_mind_prob[event_id + 1][key]

            temp = valid_pool*likelihood*event_prior
            if np.sum(temp) == 0:
                print(valid_pool, likelihood, event_prior)
            if np.sum(temp) > 0:
                temp = temp/np.sum(temp)
            mind_sep[key] = temp.reshape(-1)

        combination_list = list(product([0, 1, 2, 3], repeat = 5))
        combination_best = None
        best_prob = 0
        count = 0
        for combination in combination_list:
            com_prob = mind_sep['m1'][combination[0]]*mind_sep['m2'][combination[1]]*mind_sep['m12'][combination[2]]*mind_sep['m21'][combination[3]]*mind_sep['mc'][combination[4]]
            if not combination in self.between_mind_prob:
                continue
            count += 1
            prior = self.between_mind_prob[combination]
            if prior*com_prob >= best_prob:
                best_prob = prior*com_prob
                combination_best = combination

            #print(prior)
            # print(com_prob)

        # print(combination_best)
        # print(count)

        assert combination_best
        assert count == len(self.between_mind_prob)

        new_node['mind'] = combination_best
        new_node['valid_pool'] = {}
        mind_ids = ['m1', 'm2', 'm12', 'm21', 'mc']
        final_prob = np.array([mind_sep['m1'][combination_best[0]], mind_sep['m2'][combination_best[1]],
                               mind_sep['m12'][combination_best[2]], mind_sep['m21'][combination_best[3]], mind_sep['mc'][combination_best[4]]])

        for mind_id, change_type in enumerate(combination_best):
            if change_type == 0:
                valid_pool = np.array([0, 1, 1, 1])
                valid_pool = valid_pool*final_prob[mind_id] + (valid_pool == 0)*(1-final_prob[mind_id])
            elif change_type == 1:
                valid_pool = np.array([1, 0, 0, 1])
                valid_pool = valid_pool * final_prob[mind_id] + (valid_pool == 0) * (1 - final_prob[mind_id])
            elif change_type == 2:
                valid_pool = np.array([0, 1, 1, 1])
                valid_pool = valid_pool * final_prob[mind_id] + (valid_pool == 0) * (1 - final_prob[mind_id])
            elif change_type == 3:
                if len(self.Tree['nodes']) == 0:
                    valid_pool = np.array([1/2., 0, 0, 1/2.])
                else:
                    valid_pool = self.Tree['nodes'][-1]['valid_pool'][mind_ids[mind_id]]
                    max_value = np.max(valid_pool)
                    valid_pool[valid_pool != max_value] = 0
                    valid_pool[valid_pool == max_value] = 1
                    valid_pool = valid_pool * final_prob[mind_id] + (valid_pool == 0) * (1 - final_prob[mind_id])
            new_node['valid_pool'][mind_ids[mind_id]] = valid_pool

        self.Tree['nodes'].append(new_node)

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

    def tree_grow(self,process_i, mind_output):
        #print("I am process {}-- into tree grow".format(process_i))
        '''
        input:
            current top N possible path
        :return:
            new top N possible path
        '''
        start_time = time.time()

        new_node = {}
        mindoutput = mind_output[len(self.Tree['nodes'])]

        new_node['m1'] = mindoutput['m1']
        new_node['m2'] = mindoutput['m2']
        new_node['m12'] = mindoutput['m12']
        new_node['m21'] = mindoutput['m21']
        new_node['mc'] = mindoutput['mc']
        new_node['event'] = mindoutput['event']

        self.tree_prune(new_node)
        self.Tree['level'] += 1
        if len(self.Tree['nodes']) == self.clip_length:
            return self.Tree['nodes']
        else:
            return None

def finetune_para(mind_net, clip_list, args):

    pid = current_thread().name
    # TODO
    if args.resume:
        try:
            with open(op.join(args.save_path, 'resume_rec_para.p'), 'rb') as f:
                resume_rec=pickle.load(f)
                i_clip_sp, clip_, cnt_clip, obj_id_start = resume_rec

        except:
            print("!"*10)
            print("\033[31m ERROR: no correct resume_rec file! \033[0m")
            i_clip_sp = 0
            obj_id_start = 0
    else:
        i_clip_sp=0
        obj_id_start = 0

    args.topN=5
    args.search_N_cp=5

    for i_clip in range(i_clip_sp, len(clip_list)):
        clip = clip_list[i_clip]
        if not os.path.exists(args.annotation_path + clip.split('.')[0] + '.txt'):
            continue

        if not os.path.exists(args.save_path + '/' + clip.split('.')[0]):
            os.makedirs(args.save_path + '/' + clip.split('.')[0])

        annt = pd.read_csv(args.annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()

        for obj_id in range(obj_id_start, len(obj_names)):
            obj_name = obj_names[obj_id]
            if os.path.exists(args.save_path + '/' + clip.split('.')[0] + '/' + obj_name + '.p'):
                continue
            if obj_name.startswith('P'):
                continue
            beam_search = BeamSearch_Mind(args, mind_net)

            with open(args.mind_input_path + '/' + clip.split('.')[0] + '/' + obj_name + '.p', 'rb') as f:
                mind_output = pickle.load(f)[0]

            beam_search.init(clip)

            print(" PID {}, video {} ({}/{})".format(pid, clip, i_clip, len(clip_list)-1))
            print("="*74)

            # beam search
            while True:
                Tree_best=beam_search.tree_grow(pid, mind_output)
                if Tree_best is not None:
                    break

            with open(args.save_path + '/' + clip.split('.')[0] + '/' + obj_name + '.p', 'wb') as f:
                pickle.dump(Tree_best, f)

def key_frame(args):

    save_path = args.project_path2+'key_frame_ours_0530/'
    save_path='/home/lfan/Desktop/'+'key_frame/'
    if not op.exists(save_path):
        os.makedirs(save_path)

    for clip_ in ['test_boelter_10']:
        clip=clip_.split('.')[0]

        if not op.exists(save_path+clip+'/'):
            os.makedirs(save_path+clip+'/')

        obj_names = glob.glob(op.join(args.save_path, clip, '*.p'))
        img_names = sorted(glob.glob(op.join(args.img_path, clip, 'kinect', '*.jpg')))
        key_frame = []
        total_prob_temp = []

        for obj_name in obj_names:
            with open(obj_name, 'rb') as f:
                mind_changes = pickle.load(f)

            for frame_id, mind_change in enumerate(mind_changes):

                mind_names = ['m1', 'm2', 'm12', 'm21', 'mc']
                prod_prob = 1
                for mind_id, mind_name in enumerate(mind_names):
                    prod_prob *= np.sum(mind_change[mind_name][0][:2])
                total_prob_temp.append(prod_prob)
                key_frame.append(frame_id)
                # if prod_prob > 0.5:
                #     key_frame.append(frame_id)
        ratio = int(len(img_names)*0.01)
        print(ratio)
        total_prob_temp = np.array(total_prob_temp)
        frame_ids = np.argsort(total_prob_temp)[::-1]
        frame_ids = frame_ids[:ratio]
        print(total_prob_temp[frame_ids])

        print(total_prob_temp)

        key_frame = np.unique(np.array(key_frame)[frame_ids])
        key_frame = np.sort(key_frame)

        for key_frame_id in key_frame:
            img = cv2.imread(img_names[key_frame_id])
            cv2.imwrite(save_path + clip + '/' + img_names[key_frame_id].split('/')[-1], img)


def key_frame2(args):

    save_path = args.project_path + 'supplementary/keyframe_0610_2/'
    if not op.exists(save_path):
        os.makedirs(save_path)

    #for clip in ['test_boelter_10', 'test_94342_11', 'test_94342_15', 'test_94342_17', 'test_94342_20', 'test_94342_24', 'test_boelter2_6', 'test_boelter3_5', 'test_boelter4_7', 'test_boelter4_10', 'test_boelter4_0']:
    for clip in mind_test_clips:
        clip=clip.split('.')[0]

        if not op.exists(save_path + clip + '/'):
            os.makedirs(save_path + clip + '/')

        obj_names = glob.glob(op.join(args.save_path, clip, '*.p'))
        img_names = sorted(glob.glob(op.join(args.img_path, clip, 'kinect', '*.jpg')))
        total_prob_frames = np.zeros(len(img_names))

        for obj_name in obj_names:
            with open(obj_name, 'rb') as f:
                mind_changes = pickle.load(f)

            for frame_id, mind_change in enumerate(mind_changes):

                mind_names = ['m1', 'm2', 'm12', 'm21', 'mc']
                prod_prob = 1
                for mind_id, mind_name in enumerate(mind_names):
                    # prod_prob *= np.sum(mind_change[mind_name][0][:2])
                    prod_prob *= np.sum(mind_change[mind_name][0][2:])
                total_prob_frames[frame_id] += 1-prod_prob
                # if prod_prob > 0.5:
                #     key_frame.append(frame_id)
        ratio = int(len(img_names) * 0.01)
        print(ratio)
        total_prob_temp = np.array(total_prob_frames)
        frame_ids = np.argsort(total_prob_temp)[::-1]
        frame_ids = frame_ids[:ratio]
        frame_ids=frame_ids[:12]
        assert len(total_prob_temp) == len(img_names)
        # total_prob_temp = total_prob_temp/np.sum(total_prob_temp)
        total_prob_temp = total_prob_temp/len(obj_names)

        with open(save_path + clip + '/frame_scores.p','wb') as f:
            pickle.dump(total_prob_temp, f)

        for key_frame_id in frame_ids:
            img = cv2.imread(img_names[key_frame_id])
            cv2.imwrite(save_path + clip + '/' + img_names[key_frame_id].split('/')[-1], img)

def keyframe2gif():

    for clip in mind_test_clips:
        # clip=mind_test_clips[0]
        folder=op.join('/home/lfan/Destop/', clip.split('.')[0])
        os.system('convert -delay 70 -loop 0 '+folder+'/*.jpg '+clip.split('.')[0]+'.gif')

def draw_res():
    gt_path='/home/lfan/Dropbox/Projects/NIPS20/regenerate_annotation/test_boelter_10.p'
    with open(gt_path, 'rb') as f:
        gt_belief=pickle.load(f)
    prediction_path='/media/lfan/HDD/NIPS20/mind_search_output_0530/test_boelter_10/'
    with open(op.join(prediction_path, 'O1.p'), 'rb') as f:
        O1=pickle.load(f)
    with open(op.join(prediction_path, 'O2.p'), 'rb') as f:
        O2=pickle.load(f)

    L=len(O1)
    m1=[]
    m2=[]
    m12=[]
    m21=[]
    mc=[]
    gt=[]

    gt_m1=[]
    gt_m2=[]
    gt_m12=[]
    gt_m21=[]
    gt_mc=[]

    for t in range(L):
        # print(O1[t]['mc'])
        # m1_change=O1[t]['m1'][0, 0]+O1[t]['m1'][0, 1]+O2[t]['m1'][0, 0]+O2[t]['m1'][0, 1]
        # m1.append(m1_change/2.)
        # m2_change = O1[t]['m2'][0, 0] + O1[t]['m2'][0, 1] + O2[t]['m2'][0, 0] + O2[t]['m2'][0, 1]
        # m2.append(m2_change/2.)
        # m12_change = O1[t]['m12'][0, 0] + O1[t]['m12'][0, 1] + O2[t]['m12'][0, 0] + O2[t]['m12'][0, 1]
        # m12.append(m12_change/2.)
        # m21_change = O1[t]['m21'][0, 0] + O1[t]['m21'][0, 1] + O2[t]['m21'][0, 0] + O2[t]['m21'][0, 1]
        # m21.append(m21_change/2.)

        # if gt_belief['O1'][t]['m1']['fluent']<2 or gt_belief['O2'][t]['m1']['fluent']<2 or gt_belief['O1'][t]['m2']['fluent']<2 or gt_belief['O2'][t]['m2']['fluent']<2 \
        #         or  gt_belief['O1'][t]['m12']['fluent']<2 or gt_belief['O2'][t]['m12']['fluent']<2 or  gt_belief['O1'][t]['m21']['fluent']<2 or gt_belief['O2'][t]['m21']['fluent']<2\
        #         or  gt_belief['O1'][t]['mc']['fluent']<2 or gt_belief['O2'][t]['mc']['fluent']<2:
        #     gt_change=1
        # else:
        #     gt_change=0
        # gt.append(gt_change)
        #

        if gt_belief['O1'][t]['m1']['fluent']<2 or gt_belief['O2'][t]['m1']['fluent']<2:
            gt_m1_change=1
        else:
            gt_m1_change=0
        gt_m1.append(gt_m1_change)


        if gt_belief['O1'][t]['m2']['fluent']<2 or gt_belief['O2'][t]['m2']['fluent']<2:
            gt_m2_change=1
        else:
            gt_m2_change=0
        gt_m2.append(gt_m2_change)


        if gt_belief['O1'][t]['m12']['fluent']<2 or gt_belief['O2'][t]['m12']['fluent']<2:
            gt_m12_change=1
        else:
            gt_m12_change=0
        gt_m12.append(gt_m12_change)


        if gt_belief['O1'][t]['m21']['fluent']<2 or gt_belief['O2'][t]['m21']['fluent']<2:
            gt_m21_change=1
        else:
            gt_m21_change=0
        gt_m21.append(gt_m21_change)


        if gt_belief['O1'][t]['mc']['fluent']<2 or gt_belief['O2'][t]['mc']['fluent']<2:
            gt_mc_change=1
        else:
            gt_mc_change=0
        gt_mc.append(gt_mc_change)

    labels=['m1', 'm2', 'm12', 'm21', 'mc', 'GT']

    # plt.bar(range(len(m1)), m1,color=(0.2, 0.4, 0.6, 0.6) )
    # plt.bar(range(len(m2)), m2, color=(0.8, 0, 0.1, 0.5))
    # plt.bar(range(len(m12)), m12, color=(0.1, 0.3, 0.5, 0.4))
    # plt.bar(range(len(m21)), m21, color=(0.7, 0.5, 0.8, 0.6))
    # plt.bar(range(len(m1)), gt, color=(0.15, 0.45, 0.65, 0.6))

    # mind_dict = {'m1': m1, 'm2':m2, 'm12':m12, 'm21':m21, 'mc':[0]*len(m1), 'GT':gt}
    # df = pandas.DataFrame.from_dict(mind_dict)
    # with open('flow.p', 'wb') as f:
    #     pickle.dump(df, f)

    mind_dict = {'gt_m1': gt_m1, 'gt_m2':gt_m2, 'gt_m12':gt_m12, 'gt_m21':gt_m21, 'gt_mc':gt_mc}
    df = pandas.DataFrame.from_dict(mind_dict)
    with open('gt_five_mind_change.p', 'wb') as f:
        pickle.dump(df, f)


    # plt.plot(zip(m1), sns.xkcd_rgb["dusty purple"], lw=2)
    # plt.plot(zip(m2), sns.xkcd_rgb["amber"], lw=2)
    # plt.plot(zip(m12), sns.xkcd_rgb["pale red"], lw=2)
    # plt.plot(zip(m21), sns.xkcd_rgb["faded green"], lw=2)
    # plt.plot(zip([0]*len(m1)), sns.xkcd_rgb["greyish"], lw=2)
    # plt.plot(zip(gt), sns.xkcd_rgb["windows blue"], lw=2)
    # plt.legend(labels)

    # ours_highlight=np.zeros(len(gt))
    # ours_highlight[1]   = 1
    # ours_highlight[223] = 1
    # ours_highlight[301] = 1
    # ours_highlight[366] = 1
    # ours_highlight[560] = 1
    # ours_highlight[585] = 1
    # plt.bar(range(len(m1)), ours_highlight, width=3, color='yellow')


    # human_highlight=np.zeros(len(gt))
    # human_highlight[17]   = 1
    # human_highlight[161] = 1
    # human_highlight[234] = 1
    # human_highlight[513] = 1
    # human_highlight[556] = 1
    # human_highlight[592] = 1
    # plt.bar(range(len(m1)), human_highlight, width=3, color='yellow')

    # ddp_highlight=np.zeros(len(gt))
    # ddp_highlight[0]   = 1
    # ddp_highlight[204] = 1
    # ddp_highlight[205] = 1
    # ddp_highlight[208] = 1
    # ddp_highlight[221] = 1
    # ddp_highlight[653] = 1
    # plt.bar(range(len(m1)), ddp_highlight, width=3, color='yellow')

    # fcsn_highlight=np.zeros(len(gt))
    # fcsn_highlight[441]   = 1
    # fcsn_highlight[443] = 1
    # fcsn_highlight[446] = 1
    # fcsn_highlight[459] = 1
    # fcsn_highlight[648] = 1
    # fcsn_highlight[650] = 1
    # plt.bar(range(len(m1)), fcsn_highlight, width=2.5, color='yellow')

    # dsn_highlight=np.zeros(len(gt))
    # dsn_highlight[202]   = 1
    # dsn_highlight[205] = 1
    # dsn_highlight[206] = 1
    # dsn_highlight[207] = 1
    # dsn_highlight[303] = 1
    # dsn_highlight[304] = 1
    # plt.bar(range(len(m1)), dsn_highlight, width=2.5, color='yellow')

    plt.show()

    # plt.plot(zip(gt))
    # plt.show()
    # plt.plot(m1, '.')
    # plt.plot(m2, '.')
    # plt.plot(m12, '.')
    # plt.plot(m21, '.')

    # plt.plot(mc, '.')
    # plt.show()


def parse_arguments():

    parser=argparse.ArgumentParser(description='')
    # path
    path = Path('home')
    parser.add_argument('--project-path',default = path.home_path)
    parser.add_argument('--project-path2', default=path.home_path2)
    parser.add_argument('--data-path', default=path.data_path)
    parser.add_argument('--data-path2', default=path.data_path2)
    parser.add_argument('--img-path', default=path.img_path)
    parser.add_argument('--annotation-path', default=path.annotation_path)
    parser.add_argument('--save-root', default=path.save_root)
    parser.add_argument('--save-path', default=path.save_root + '/mind_search_output_0530/')
    parser.add_argument('--mind-input-path', default=path.save_root + '/mind_search_input_0530/')
    parser.add_argument('--mind-model-path', default=path.mind_model_path)
    # others
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--resume', default=False)

    return parser.parse_args()


def mind_stat():

    args=parse_arguments()

    with open(op.join(args.data_path, 'event_mind_prob.p'), 'rb') as f:
        event_mind_prob = pickle.load(f)

    for i in range(1, len(event_mind_prob)):
        count = []
        for key in event_mind_prob[i]:
            mind_count = np.array(
                [event_mind_prob[i][key][0], event_mind_prob[i][key][1], event_mind_prob[i][key][2],
                 event_mind_prob[i][key][3]])
            # mind_count = mind_count / float(np.sum(mind_count))
            # assert abs(np.sum(mind_count) - 1) < 0.001
            event_mind_prob[i][key] = mind_count

    with open(op.join(args.data_path, 'between_mind_prob.p'), 'rb') as f:
        between_mind_prob = pickle.load(f)

    count = 0
    for key in between_mind_prob.keys():
        count += between_mind_prob[key]

    count_list = np.empty(0)
    for key in between_mind_prob.keys():
        between_mind_prob[key] = between_mind_prob[key] / float(count)
        count_list = np.append(count_list, between_mind_prob[key])

    pass


if __name__ == "__main__":

    # print("=" * 74)
    # print('/' * 5)
    # print("/ \033[31m [Important!] Don't forget to change your project name accordingly! \033[0m ")
    # print('/' * 5)
    # print("=" * 74)
    #
    # args = parse_arguments()
    # if not op.exists(args.save_path):
    #     os.makedirs(args.save_path)
    #
    # mind_net = MindHog()
    # mind_net.load_state_dict(torch.load(args.mind_model_path))
    # if args.cuda and torch.cuda.is_available():
    #     mind_net.cuda()
    # mind_net.eval()
    #
    # random.seed(0)
    # random.shuffle(clips_all)
    #
    # Parallel(n_jobs=1)(delayed(finetune_para)(mind_net, [clip], args) for clip in mind_test_clips)
    #
    # args=parse_arguments()
    # key_frame(args)
    #
    # keyframe2gif()

    # draw_res()

    #
    # beam=BeamSearch_Mind(args, mind_net)
    # beam.init(['test1.p'])

    args=parse_arguments()
    key_frame2(args)

    # mind_stat()

