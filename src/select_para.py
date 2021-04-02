import numpy as np
import os.path as op
import pickle
import glob
from beam_search_0907 import parse_arguments

def select_para(args):

    path=args.save_path

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

    N=5
    print('seg')
    print(seg_rank_[:N])
    print(para_bank[seg_idxrank[:N]])
    print('event')
    print(event_rank_[:N])
    print(para_bank[event_idxrank[:N]])
    print('all')
    print(all_rank_[:N])
    print(para_bank[all_idxrank[:N]])


if __name__ == "__main__":

    args=parse_arguments()

    select_para(args)