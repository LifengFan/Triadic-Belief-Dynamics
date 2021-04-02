from utils import *
import os.path
from  Atomic_node_only_lstm_517 import  Atomic_node_only_lstm_first_view
from joblib import Parallel, delayed
from para_bank import *
from beam_search_class_0601 import *

def finetune_para_E(atomic_net, event_net, i_clip, clip, para_idx, para, args):

    beam_search = BeamSearch_E(args, event_net, atomic_net)

    args.topN=para['topN']
    args.lambda_1 = para['lambda_1']
    args.lambda_4=para['lambda_4']
    args.lambda_5=para['lambda_5']
    args.beta_1=para['beta_1']
    args.beta_3=para['beta_3']
    args.gamma_1=para['gamma_1']
    args.search_N_cp=para['search_N_cp']

    with open(op.join(args.tree_best_P_path, clip), 'rb') as f:
        Tree_best_P=pickle.load(f)

    beam_search.init(clip, Tree_best_P)

    while True:
        Tree_best=beam_search.tree_grow()
        if Tree_best is not None:
            break

    # evaluation
    cps_T=Tree_best['T']['cp']
    cps_B=Tree_best['B']['cp']
    event_T=Tree_best['T']['event']
    event_B=Tree_best['B']['event']
    # seg
    segs_T=beam_search.cps2segs(cps_T)
    segs_B=beam_search.cps2segs(cps_B)

    # event
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

    with open(op.join(args.save_path, "E_para_err_clip_{}_para_{}.p".format(i_clip, para_idx)), 'wb') as f:
        pickle.dump([clip, para, err_event, Tree_best], f)


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
    parser.add_argument('--save-path', default='/media/lfan/HDD/NIPS20/Result2/BeamSearch_best_para_home_0531_1/')
    parser.add_argument('--init-cps', default='/media/lfan/HDD/NIPS20/data/init_cps/CPS_NEW.p')
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
    parser.add_argument('--save-event-score', default='/media/lfan/HDD/NIPS20/Result/EVENT_SCORE/')
    parser.add_argument('--tree-best-P-path', default='/media/lfan/HDD/NIPS20/Result0601/Finetune_P_home_v1/Tree_best/')

    # parameter
    parser.add_argument('--lambda-1', default=1)
    parser.add_argument('--lambda-4', default=10)
    parser.add_argument('--lambda-5', default=1)
    parser.add_argument('--beta-1', default=1)
    parser.add_argument('--beta-3', default=10)
    parser.add_argument('--gamma-1', default=10)
    parser.add_argument('--search-N-cp', default=5)
    parser.add_argument('--topN', default=5)

    parser.add_argument('--hist-bin', default=10)
    parser.add_argument('--seg-alpha', default=50)

    # others
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--ip', default='192.168.1.17')
    parser.add_argument('--port', default=1234)
    parser.add_argument('--resume',default=False)
    parser.add_argument('--test-func-batch-size', default=16)

    return parser.parse_args()


def select_para():

    path='/media/lfan/HDD/NIPS20/Result0601'

    seg_rank=[]
    event_rank=[]
    all_rank=[]

    with open(op.join(path, 'para_bank.p'), 'rb') as f:
        para_bank=pickle.load(f)

    files=glob.glob(op.join(path, '*.p'))

    para_list=[]
    for file in files:
        name=file.split('/')[-1]
        if name=='para_bank.p':
            continue
        else:
            para_list.append(name.split('_')[2])

    para_list=list(np.unique(np.array(para_list)))

    for para_idx in para_list:
        seg_err = []
        event_err = []
        all_err = []

        files=glob.glob(op.join(path, 'RES_para_{}_*'.format(para_idx)))
        for file in files:
            with open(file, 'rb') as f:
                para, clip, Tree_best, err_seg, err_event=pickle.load(f)

                seg_err.append(err_seg)
                event_err.append(err_event)
                all_err.append(err_seg+err_event)

        seg_mean=np.mean(np.array(seg_err))
        event_mean=np.mean(np.array(event_err))
        all_mean=np.mean(np.array(all_err))

        seg_rank.append(seg_mean)
        event_rank.append(event_mean)
        all_rank.append(all_mean)

    seg_rank_=list(np.sort(np.array(seg_rank)))
    event_rank_=list(np.sort(np.array(event_rank)))
    all_rank_=list(np.sort(np.array(all_rank)))

    seg_idxrank=list(np.argsort(np.array(seg_rank)))
    event_idxrank=list(np.argsort(np.array(event_rank)))
    all_idxrank=list(np.argsort(np.array(all_rank)))

    para_bank=np.array(para_bank)

    N=3
    print('seg')
    print(seg_rank_[:N])
    print(para_bank[seg_idxrank[:N]])
    print('event')
    print(event_rank_[:N])
    print(para_bank[event_idxrank[:N]])
    print('all')
    print(all_rank_[:N])
    print(para_bank[all_idxrank[:N]])


def run_finetune_para_E(args):

    args = parse_arguments()
    args.save_path='/media/lfan/HDD/NIPS20/Result0602/Finetune_E_home_v1/'
    if not op.exists(args.save_path):
        os.makedirs(args.save_path)

    atomic_event_net = Atomic_node_only_lstm_first_view()
    load_best_checkpoint(atomic_event_net, path=args.atomic_path)
    if args.cuda and torch.cuda.is_available():
        atomic_event_net.cuda()
    atomic_event_net.eval()

    event_net=EDNet()
    event_net.load_state_dict(torch.load(args.ednet_path))
    if args.cuda and torch.cuda.is_available():
        event_net.cuda()
    event_net.eval()

    para_bank = generate_para_bank_E(args)

    random.seed(0)
    random.shuffle(clips_with_gt_event)

    para_all=para_bank
    clips_all=clips_with_gt_event

    combinations=list(product(range(len(para_all)), range(len(clips_all))))
    print('There are {} tasks in total. {} clips, {} paras'.format(len(combinations), len(clips_all), len(para_all)))
    Parallel(n_jobs=-1)(delayed(finetune_para_E)(atomic_event_net, event_net, comb[1], clips_all[comb[1]], comb[0], para_all[comb[0]], args) for _,  comb in enumerate(combinations))

if __name__ == "__main__":

    # run_finetune_para()

    select_para()




