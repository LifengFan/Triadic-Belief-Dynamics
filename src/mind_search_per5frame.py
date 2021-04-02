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
    parser.add_argument('--save-path', default=path.save_root + '/mind_posterior_per5frame_0531/')
    parser.add_argument('--mind-input-path', default=path.save_root + '/mind_likelihood_per5frame_0531/')
    parser.add_argument('--mind-model-path', default=path.mind_model_path)
    # others
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--resume', default=False)

    return parser.parse_args()

if __name__ == "__main__":

    print("=" * 74)
    print('/' * 5)
    print("/ \033[31m [Important!] Don't forget to change your project name accordingly! \033[0m ")
    print('/' * 5)
    print("=" * 74)

    args = parse_arguments()
    if not op.exists(args.save_path):
        os.makedirs(args.save_path)

    mind_net = MindHog()
    mind_net.load_state_dict(torch.load(args.mind_model_path))
    if args.cuda and torch.cuda.is_available():
        mind_net.cuda()
    mind_net.eval()

    random.seed(0)
    random.shuffle(clips_all)

    Parallel(n_jobs=1)(delayed(finetune_para)(mind_net, [clip], args) for clip in mind_test_clips)

    # args=parse_arguments()
    # key_frame(args)

    # keyframe2gif()