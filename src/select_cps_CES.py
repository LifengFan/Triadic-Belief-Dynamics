
import pickle
from metadata import *
from utils import *
import argparse
import numpy as np


def parse_arguments():

    parser=argparse.ArgumentParser(description='generate_seg_points')
    parser.add_argument('--project-path',default = '/home/lfan/Dropbox/Projects/NIPS20/')
    parser.add_argument('--data-path', default='/home/lfan/Dropbox/Projects/NIPS20/data/')
    parser.add_argument('--data-path2', default='/media/lfan/HDD/NIPS20/data/')
    parser.add_argument('--res-path', default='/media/lfan/HDD/NIPS20/result_init_cps/')
    parser.add_argument('--save-path', default='/home/lfan/Dropbox/Projects/NIPS20/result/')
    parser.add_argument('--seg-method')
    parser.add_argument('--cps-path', default='/media/lfan/HDD/NIPS20/data/init_cps/')
    parser.add_argument('--single-feature-path', default='/home/lfan/Dropbox/Projects/NIPS20/data/feature_single/')
    parser.add_argument('--pair-feature-path', default='/home/lfan/Dropbox/Projects/NIPS20/data/feature_pair/')
    parser.add_argument('--clips', default=os.listdir('/home/lfan/Dropbox/Projects/NIPS20/data/check_points/'))
    parser.add_argument('--img-path', default='/home/lfan/Dropbox/Projects/NIPS20/annotations/')
    parser.add_argument('--concate', default=False)
    parser.add_argument('--concate-N', default=4)
    parser.add_argument('--err-thr', default=20)

    return parser.parse_args()



if __name__ == '__main__':

    args=parse_arguments()
    path=Path('home')

    with open(path.home_path+'code/tracker_prediction.p', 'rb') as f:
        T=pickle.load(f, encoding='latin1')

    with open(path.home_path+'code/battery_prediction.p', 'rb') as f:
        B=pickle.load(f, encoding='latin1')

    T_N={}
    B_N={}

    for clip in clips_all:
        clip=clip.split('.p')[0]
        np.append(T[clip],[0, clips_len[clip+'.p']])
        np.append(B[clip], [0, clips_len[clip+'.p']])
        T[clip] = sorted(np.unique(np.array(T[clip])))
        B[clip] = sorted(np.unique(np.array(B[clip])))
        T_N[clip + '.p'] = T[clip]
        B_N[clip + '.p'] = B[clip]

    with open(os.path.join(args.cps_path, 'CPS_CES_0908.p'), 'wb') as f:
        pickle.dump([T_N, B_N], f)


