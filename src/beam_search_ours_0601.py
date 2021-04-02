from metadata import *
from utils import *
from overall_event_get_input import *
import time
import threading
import os, os.path
from metric import *
from  Atomic_node_only_lstm_517 import  Atomic_node_only_lstm_first_view
from joblib import Parallel, delayed
from para_bank import generate_para_bank
from beam_search_class_0601 import *


def test_best_Tree(beam_search_P, beam_search_E, clip, args):

    pid = threading.get_ident()
    print('pid: {} clip {}'.format(pid, clip))

    beam_search_P.init(clip)
    while True:
        Tree_best=beam_search_P.tree_grow()
        if Tree_best is not None:
            break

    beam_search_E.init(clip, Tree_best)
    while True:
        Tree_best=beam_search_E.tree_grow()
        if Tree_best is not None:
            break

    with open(op.join(args.save_path, clip), 'wb') as f:
        pickle.dump(Tree_best, f, protocol=2)

def finetune_para_P(atomic_net, event_net, clip_list, para_idx, para_bank, args):

    beam_search = BeamSearch_P(args, event_net, atomic_net)
    start_time=time.time()
    pid = threading.get_ident()

    with open(op.join(args.save_path, 'resume_rec_para_{}.p'.format(para_idx)), 'rb') as f:
        resume_rec=pickle.load(f, encoding='latin1')
        para_idx_, i_clip_sp, para_, clip_, cnt_clip, err_seg, err_event=resume_rec
        assert para_idx_== para_idx
        i_para_sp = 0
        ERR_PARA = []

    for i_para in range(len(para_bank)):
        para=para_bank[i_para]

        args.topN=para['topN']
        args.lambda_1 = para['lambda_1']
        args.lambda_4=para['lambda_4']
        args.lambda_5=para['lambda_5']
        args.beta_1=para['beta_1']
        args.beta_3=para['beta_3']
        args.gamma_1=para['gamma_1']
        args.search_N_cp=para['search_N_cp']

        finetune_save_path=args.save_path

        if i_para>i_para_sp:
            i_clip_sp=0
        for i_clip in range(i_clip_sp, len(clip_list)):
            if i_clip==0:
                cnt_clip = 0
                err_seg = 0
                err_event = 0

            clip=clip_list[i_clip]

            with open(op.join(args.save_path, 'resume_rec_para_{}.p'.format(para_idx)), 'wb') as f:
                pickle.dump([para_idx, i_clip,  para, clip, cnt_clip, err_seg, err_event], f)

            beam_search.init(clip)

            print(" PID {}, para_idx {}, video {} ({}/{})".format(pid, para_idx, clip, i_clip, len(clip_list)-1))
            print("="*74)

            # beam search
            while True:
                Tree_best=beam_search.tree_grow(pid)
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
            tracker_gt_seg = event_seg_tracker[clip.split('.')[0]]
            battery_gt_seg = event_seg_battery[clip.split('.')[0]]

            err_seg += segment_error(segs_T, tracker_gt_seg, args.seg_alpha) + segment_error(segs_B, battery_gt_seg,args.seg_alpha)

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

            err_event += np.sum(frame_events_T != frame_events_T_gt) + np.sum(frame_events_B != frame_events_B_gt)
            cnt_clip += 1

            print("pid {} para_idx {} vid {} temp_err {} used time {}".format(pid, para_idx, i_clip, (err_seg + err_event) / float(cnt_clip), time.time() - start_time))
            print("="*70)

        # current para
        assert(cnt_clip>0)
        ERR_PARA.append((err_seg + err_event)/float(cnt_clip))
        print("pid {} para_idx {} err {} used time {}".format(pid, para_idx, (err_seg + err_event)/float(cnt_clip), time.time()-start_time))
        print("=" * 70)


        with open(op.join(finetune_save_path, "para_err_{}.p".format(para_idx)), 'wb') as f:
            pickle.dump([para_idx, para,(err_seg + err_event)/float(cnt_clip), Tree_best, clip], f)

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


def run_finetune_para():

    args = parse_arguments()
    args.save_path='/media/lfan/HDD/NIPS20/Result0531/BeamSearch_best_para_home_0531_1/'
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

    para_bank = generate_para_bank(args)

    random.seed(0)
    random.shuffle(clips_with_gt_event)

    para_all=para_bank
    clips_all=clips_with_gt_event

    combinations=list(product(range(len(para_all)), range(len(clips_all))))
    print('There are {} tasks in total. {} clips, {} paras'.format(len(combinations), len(clips_all), len(para_all)))
    Parallel(n_jobs=-1)(delayed(finetune_para_P)(atomic_event_net, event_net, comb[1], clips_all[comb[1]], comb[0], para_all[comb[0]], args, len(clips_all), len(para_all)) for _,  comb in enumerate(combinations[7500:]))


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

    args=parse_arguments()
    args.save_path=op.join(args.project_path2, 'BestTree_ours_split_steps_0601')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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

    args.topN = 3
    args.lambda_1 = 3
    args.lambda_4 = 20
    args.lambda_5 = 30
    args.beta_1 = 13
    args.beta_3 = 0.7
    args.gamma_1 = 30
    args.search_N_cp = 5



    beam_search_P = BeamSearch_P(args, event_net, atomic_net)
    beam_search_E=BeamSearch_E(args, event_net, atomic_net)

    Parallel(n_jobs=-1)(delayed(finetune_para_P)(beam_search_P, beam_search_E, clip, args) for clip_idx, clip in enumerate(clips_all))

