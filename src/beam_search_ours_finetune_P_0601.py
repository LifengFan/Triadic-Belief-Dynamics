import threading
from joblib import Parallel, delayed
from beam_search_class_0601 import *
from para_bank import *

def finetune_para_P(i_clip, clip, i_para, para, args):

    beam_search = BeamSearch_P(args)
    pid = threading.get_ident()

    args.topN=para['topN']
    args.beta_1=para['beta_1']
    args.beta_3=para['beta_3']
    args.search_N_cp=para['search_N_cp']

    beam_search.init(clip)

    print(" PID {}, para_idx {}, video {}".format(pid, i_para, clip))
    print("="*74)

    # beam search
    while True:
        Tree_best=beam_search.tree_grow()
        if Tree_best is not None:
            break

    # evaluation
    cps_T=Tree_best['T']['cp']
    cps_B=Tree_best['B']['cp']

    # seg
    segs_T=beam_search.cps2segs(cps_T)
    segs_B=beam_search.cps2segs(cps_B)
    tracker_gt_seg = event_seg_tracker[clip.split('.')[0]]
    battery_gt_seg = event_seg_battery[clip.split('.')[0]]

    err_seg = segment_error(segs_T, tracker_gt_seg, args.seg_alpha) + segment_error(segs_B, battery_gt_seg,args.seg_alpha)

    with open(op.join(args.save_path, "P_para_err_clip_{}_para_{}.p".format(i_clip, i_para)), 'wb') as f:
        pickle.dump([clip, para, err_seg, Tree_best], f)

def select_para(args, para_bank, N=3):

    path=args.save_path

    seg_rank=[]

    files=glob.glob(op.join(path, '*.p'))

    para_list=[]
    for file in files:
        name=file.split('/')[-1]
        if name=='para_bank.p':
            continue
        else:
            para_list.append(name.split('_')[-1].split('.')[0])

    para_list=list(np.unique(np.array(para_list)))

    for para_idx in para_list:
        seg_err = []

        files=glob.glob(op.join(path, 'P_para_err_clip_*_para_{}.p'.format(para_idx)))
        for file in files:
            with open(file, 'rb') as f:
                clip, para, err_seg, Tree_best=pickle.load(f)

                seg_err.append(err_seg)

        seg_mean=np.mean(np.array(seg_err))
        seg_rank.append(seg_mean)

    seg_rank_=list(np.sort(np.array(seg_rank)))
    seg_idxrank=list(np.argsort(np.array(seg_rank)))
    para_bank=np.array(para_bank)
    print('seg')
    print(seg_rank_[:N])
    print(para_bank[seg_idxrank[:N]])

def test_best_Tree_P(clip, args):

    pid = threading.get_ident()
    print('pid: {} clip {}'.format(pid, clip))

    beam_search = BeamSearch_P(args)

    beam_search.init(clip)
    while True:
        Tree_best=beam_search.tree_grow()
        if Tree_best is not None:
            break

    with open(op.join(args.tree_best_P_path, clip), 'wb') as f:
        pickle.dump(Tree_best, f, protocol=2)


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
    parser.add_argument('--init-cps', default='/media/lfan/HDD/NIPS20/data/init_cps/CPS_NEW_0601.p')
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
    parser.add_argument('--save-event-score', default='/media/lfan/HDD/NIPS20/data/event_score_all/')
    parser.add_argument('--tree-best-P-path', default='/media/lfan/HDD/NIPS20/Result0601/Finetune_P_home_v1/Tree_best/')
    # parameter
    parser.add_argument('--lambda-1', default=5)
    parser.add_argument('--lambda-4', default=10)
    parser.add_argument('--lambda-5', default=20)
    parser.add_argument('--beta-1', default=10)
    parser.add_argument('--beta-3', default=0.5)
    parser.add_argument('--gamma-1', default=15)
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

    args = parse_arguments()

#------------------------------------------------
# finetune
#     args.save_path='/media/lfan/HDD/NIPS20/Result0601/Finetune_P_home_v1/'
#     if not op.exists(args.save_path):
#         os.makedirs(args.save_path)
#
#     para_bank = generate_para_bank_P(args)
    #
    # random.seed(0)
    # random.shuffle(clips_with_gt_event)
    #
    # para_all=para_bank
    # clips_all=clips_with_gt_event
    #
    # combinations=list(product(range(len(para_all)), range(len(clips_all))))
    # print('There are {} tasks in total. {} clips, {} paras'.format(len(combinations), len(clips_all), len(para_all)))
    # Parallel(n_jobs=-1)(delayed(finetune_para_P)(comb[1], clips_all[comb[1]], comb[0], para_all[comb[0]], args) for _,  comb in enumerate(combinations))

#---------------------------------------------------
# select para

    # select_para(args, para_bank)

#------------------------------------------------
    # test
    random.seed(0)
    random.shuffle(clips_with_gt_event)
    args.topN=5
    args.beta_1=1
    args.beta_3=1
    args.search_N_cp=5

    Parallel(n_jobs=-1)(delayed(test_best_Tree_P)(clip, args) for _,  clip in enumerate(clips_with_gt_event))


