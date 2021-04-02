import os
import glob
import pickle
import  sys
from metadata import *
import torch
from utils import *
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
import hdbscan
import matplotlib.pyplot as plt
import argparse
import numpy as np, scipy.io
from cpd_nonlin import cpd_nonlin
from cpd_auto import cpd_auto
from metric import segment_error

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(381, 50)
        self.fc2 = torch.nn.Linear(50, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        embedding = self.fc1(x)
        out = self.relu(embedding)
        out = self.fc2(out)
        return out, embedding

def id2labels(idx, length):
    labels = []
    labels.append([0])
    counter = 0
    for i in range(1, length):
        if idx[i] != idx[i - 1]:
            labels[counter].extend([i - 1, idx[i - 1]])
            labels.append([i])
            counter += 1
    if len(labels[counter]) < 2:
        labels[counter].extend([length - 1, idx[length - 1]])
    return labels

def find_parent_id(seg_id, seg_mark):
    while seg_mark[seg_id]:
        seg_id = seg_mark[seg_id]
    return seg_id

def concate_labels(labels):
    seg_mark = {}
    for seg_id, seg in enumerate(labels):
        if seg[1] - seg[0] < 10:
            if seg_id - 1 not in seg_mark:
                seg_mark[seg_id] = None
            if seg_id - 1 in seg_mark:
                seg_mark[seg_id] = seg_id - 1

    seg_id = len(labels) - 1
    to_remove = []
    while (seg_id > 0):
        if seg_id in seg_mark:
            parent_id = find_parent_id(seg_id, seg_mark)

            if parent_id != seg_id:
                labels[parent_id][1] = labels[seg_id][1]
                if labels[parent_id][1] - labels[parent_id][0] < 10:
                    labels[parent_id - 1][1] = labels[parent_id][1]
                    for i in np.arange(seg_id, parent_id - 1, -1):
                        to_remove.append(i)
                else:
                    for i in np.arange(seg_id, parent_id, -1):
                        to_remove.append(i)
            else:
                labels[seg_id - 1][1] = labels[seg_id][1]
                to_remove.append(seg_id)
            seg_id = parent_id - 1
        else:
            seg_id = seg_id - 1

    for i in range(len(to_remove)):
        del labels[to_remove[i]]

    # ## added by Lifeng
    # if (labels[0][1]-labels[0][0])<10:
    #     labels[1][0]=labels[0][0]
    #     del labels[0]

    return labels


def sub_segment_single(second_level_seg, seg, features):
    cluster_num = 2
    start, end = seg[:2]

    hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=210).fit_predict(features[start:end])
    sub_labels = id2labels(hdbscan_cluster, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_hdscan = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_hdscan.append(start + start_sub)


    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10, max_iter=100).fit(features[start:end])
    sub_labels = id2labels(kmeans.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_kmeans = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_kmeans.append(start + start_sub)

    kernel_kmeans = SpectralClustering(n_clusters=cluster_num, random_state=0, eigen_tol=1e-15).fit(
        features[start:end])
    sub_labels = id2labels(kernel_kmeans.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_kernel_kmeans = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_kernel_kmeans.append(start + start_sub)

    agg = AgglomerativeClustering(n_clusters=cluster_num).fit(features[start:end])
    sub_labels = id2labels(agg.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_agg = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_agg.append(start + start_sub)

    dbscan32 = DBSCAN(eps=3, min_samples=2).fit(features[start:end])
    sub_labels = id2labels(dbscan32.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_dbscan32 = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_dbscan32.append(start + start_sub)

    dbscan23 = DBSCAN(eps=2, min_samples=3).fit(features[start:end])
    sub_labels = id2labels(dbscan23.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_dbscan23 = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_dbscan23.append(start + start_sub)

    dbscan55 = DBSCAN(eps=0.5, min_samples=5).fit(features[start:end])
    sub_labels = id2labels(dbscan55.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_dbscan55 = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_dbscan55.append(start + start_sub)


    return sub_check_points_kmeans, sub_check_points_kernel_kmeans, sub_check_points_agg, sub_check_points_dbscan32, \
           sub_check_points_dbscan23, sub_check_points_dbscan55
    #return sub_check_points_hdscan


def sub_segment_pair(second_level_battery_seg, second_level_tracker_seg, seg, features, args):
    cluster_num = 2
    start, end = seg[:2]

    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10, max_iter=100).fit(features[start:end])
    sub_labels = id2labels(kmeans.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_kmeans = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_kmeans.append(start + start_sub)

    kernel_kmeans = SpectralClustering(n_clusters=cluster_num, random_state=0, eigen_tol=1e-15).fit(
        features[start:end])
    sub_labels = id2labels(kernel_kmeans.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_kernel_kmeans = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_kernel_kmeans.append(start + start_sub)

    agg = AgglomerativeClustering(n_clusters=cluster_num).fit(features[start:end])
    sub_labels = id2labels(agg.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_agg = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_agg.append(start + start_sub)

    dbscan32 = DBSCAN(eps=3, min_samples=2).fit(features[start:end])
    sub_labels = id2labels(dbscan32.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_dbscan32 = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_dbscan32.append(start + start_sub)

    dbscan23 = DBSCAN(eps=2, min_samples=3).fit(features[start:end])
    sub_labels = id2labels(dbscan23.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_dbscan23 = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_dbscan23.append(start + start_sub)

    dbscan55 = DBSCAN(eps=0.5, min_samples=5).fit(features[start:end])
    sub_labels = id2labels(dbscan55.labels_, end - start)
    sub_labels = concate_labels(sub_labels)
    sub_check_points_dbscan55 = []
    for sub_id, sub_label in enumerate(sub_labels):
        start_sub, end_sub = sub_label[:2]
        sub_check_points_dbscan55.append(start + start_sub)

    return sub_check_points_kmeans, sub_check_points_kernel_kmeans, sub_check_points_agg, sub_check_points_dbscan32, \
           sub_check_points_dbscan23, sub_check_points_dbscan55

    # hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=210).fit_predict(features[start:end])
    # sub_labels = id2labels(hdbscan_cluster, end - start)
    # sub_labels = concate_labels(sub_labels)
    # sub_check_points_hdscan = []
    # for sub_id, sub_label in enumerate(sub_labels):
    #     start_sub, end_sub = sub_label[:2]
    #     sub_check_points_hdscan.append(start + start_sub)
    # return sub_check_points_hdscan

def segment(img_names, pair_features, single_features_battery, single_features_tracker, clip, net, args):

    total_labels = {}
    # joint or individual
    input = torch.from_numpy(pair_features).float().cuda()
    predicted_val, embedding = net(torch.autograd.Variable(input))

    predicted_val = predicted_val.data
    max_score, idx = torch.max(predicted_val, 1)
    idx = idx.cpu().numpy()
    labels = id2labels(idx, pair_features.shape[0])

    # pair_cps=seg2cp(labels)
    # return pair_cps

    first_level_seg = np.zeros(len(img_names))
    for seg_id, seg in enumerate(labels):
        first_level_seg[seg[0]:seg[1] + 1] = seg[2]

    total_labels['level1'] = first_level_seg

    second_level_tracker_seg = np.zeros(len(img_names))
    second_level_battery_seg = np.zeros(len(img_names))
    # check_points_tracker = {'k-means':[], 'agg':[], 'dbscan32':[], 'kernel_kmeans':[], 'dbscan23':[], 'dbscan55':[]}
    # check_points_battery = {'k-means':[], 'agg':[], 'dbscan32':[], 'kernel_kmeans':[], 'dbscan23':[], 'dbscan55':[]}
    check_points_tracker = {'hdbscan': []}
    check_points_battery = {'hdbscan': []}
    for seg_id, seg in enumerate(labels):

        # length < 10
        if seg[1] - seg[0] < 10:
            check_points_tracker['hdbscan'].append(seg[0])
            check_points_battery['hdbscan'].append(seg[0])
            continue

        # length > 10
        if seg[2] == 0:

            sub_check_points_hdbsan = sub_segment_single(second_level_battery_seg, seg, single_features_battery)
            check_points_battery['hdbscan'].extend(sub_check_points_hdbsan)
            sub_check_points_hdbsan = sub_segment_single(second_level_tracker_seg, seg, single_features_tracker)
            check_points_tracker['hdbscan'].extend(sub_check_points_hdbsan)

        else:
            sub_check_points_hdbsan = sub_segment_pair(second_level_battery_seg, second_level_tracker_seg, seg, pair_features, args)
            check_points_battery['hdbscan'].extend(sub_check_points_hdbsan)

    return check_points_tracker,check_points_battery

def cps2segs(cps):
    segs = []
    if len(cps) >= 2:
        cp_l = cps[0]
        segs = []
        for idx in range(1, len(cps)):
            cp_r = cps[idx]
            segs.append([cp_l, cp_r])
            cp_l = cp_r
    return segs

def KTS(args):

    clips=sorted(os.listdir(os.path.join(args.data_path, 'feature_single')))

    seg_points_tracker = {}
    seg_points_battery = {}

    for clip in clips:

        single_features = pickle.load(open(os.path.join(args.data_path, 'feature_single', clip)))
        clip=clip.split('.')[0]
        print(clip)

        for i in range(1, 3):
            if str(i) == tracker_skeID[clip].split('.')[0][-1]:
                single_features_tracker = single_features[i]
            else:
                single_features_battery = single_features[i]

        # tracker
        X=single_features_tracker
        # Each dimension is standardized within a video to have zero mean and unit variance. Then
        # we apply signed square-rooting and L2 normalization. We use dot products to
        # compare Fisher vectors and produce the kernel matrix.
        mu=np.mean(X, axis=0)
        X=(X-mu)/np.linalg.norm(X, axis=0)
        X=np.sign(X)*(np.sqrt(np.abs(X)))
        #X=X/np.linalg.norm(X)

        K = np.dot(X, X.T)
        # cps, scores = cpd_nonlin(K, m, lmin=1, lmax=10000)
        cps_tracker, scores = cpd_auto(K, 15, 1)
        seg_points_tracker[clip]=cps_tracker

        # plt.figure("KTS segment tracker")
        # plt.plot(X)
        # print "Estimated:", cps_tracker
        # mi = np.min(X)
        # ma = np.max(X)
        # for cp in cps_tracker:
        #     plt.plot([cp, cp], [mi, ma], 'r')
        # plt.show()
        # print "="*79

        # battery
        X=single_features_battery
        K = np.dot(X, X.T)
        # cps, scores = cpd_nonlin(K, m, lmin=1, lmax=10000)
        cps_battery, scores = cpd_auto(K, 15, 1)
        seg_points_battery[clip]=cps_battery

    save_path=args.save_path+'/KTS/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'/KTS_seg_points.p', 'wb') as f:
        pickle.dump([seg_points_tracker, seg_points_battery], f)

def HACA(args):
    # convert .pkl feature file into .mat
    save_path=args.save_path+'/singlefeature_p2mat/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clips=sorted(os.listdir(os.path.join(args.data_path, 'feature_single')))

    save_path=args.save_path+'/HACA/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seg_points_tracker = {}
    seg_points_battery = {}
    for clip in clips:

        seg_points_tracker[clip]=scipy.io.loadmat(os.path.join(save_path,clip.split('.')[0]+'_tracker'+'.mat'))
        seg_points_battery[clip]=scipy.io.loadmat(os.path.join(save_path,clip.split('.')[0]+'_battery'+'.mat'))

    with open(save_path+'/HACA_seg_points.p', 'wb') as f:
        pickle.dump([seg_points_tracker, seg_points_battery], f)

def generate_check_point(args):

    # load model
    net = MLP()
    net.load_state_dict(torch.load('./model_490.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    #pair_cps_all={}
    for clip in args.clips:
        img_names = sorted(glob.glob(os.path.join(args.img_path, clip, 'kinect/*.jpg')))
        if not os.path.exists(os.path.join(args.pair_feature_path, clip + '.p')):
            continue
        if not os.path.exists(os.path.join(args.single_feature_path, clip + '.p')):
            continue
        print(clip)
        with open(os.path.join(args.pair_feature_path, clip + '.p'), 'rb') as f:
            pair_features = pickle.load(f)
        with open(os.path.join(args.single_feature_path, clip + '.p'), 'rb') as f:
            single_features = pickle.load(f)

        for i in range(1, 3):
            if str(i) == tracker_skeID[clip].split('.')[0][-1]:
                single_features_tracker = single_features[i]
            else:
                single_features_battery = single_features[i]

        check_points_tracker,check_points_battery = segment(img_names, pair_features, single_features_battery, single_features_tracker,clip, net, args)


def seg_vis(cps, X, gt, name, args):
    plt.figure("")
    plt.plot(X)
    print(args.seg_method, "  Estimated:", cps)
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
      plt.plot([cp, cp], [mi-1, ma+1], 'r')
    for cp in gt:
      plt.plot([cp, cp], [mi-1, 0], 'g*-')
    plt.title(name)
    #plt.show()
    print("="*79)
    plt.savefig(os.path.join(args.save_path, name+'.png'), dpi=300, bbox_inches='tight')
    plt.close()


def concate_check_points_less_than_N(check_points, args):

    if len(check_points)<=2:
        return check_points
    else:
        new_check_points = []
        new_check_points.append(check_points[0])
        cnt = 0
        for i in range(1, len(check_points)):
            if check_points[i] - new_check_points[cnt] < args.concate_N:
                continue
            else:
                new_check_points.append(check_points[i])
                cnt += 1
        return new_check_points

def seg2cp(segs):
    cp = []
    for seg in segs:
        cp.append(seg[0])
    return cp

def select_cps(args):

    err_T=0
    err_B=0
    cps_comb1_T={}
    cps_comb1_B = {}

    for clip in clips_all:
        with open(os.path.join(args.single_feature_path, clip), 'rb') as f:
            single_features = pickle.load(f)

        if  args.seg_method=='hdbscan':
            with open(args.data_path + 'check_points_hdbscan/' + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T = check_points_tracker[args.seg_method]
            cps_B = check_points_battery[args.seg_method]
        elif args.seg_method=='HACA' :
            with open(args.data_path + 'HACA_seg_points.p', 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T=check_points_tracker[clip]['s_top']
            cps_B=check_points_battery[clip]['s_top']
        elif args.seg_method=='KTS':
            with open(args.data_path + 'KTS_seg_points.p', 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T=check_points_tracker[clip.split('.')[0]]
            cps_B=check_points_battery[clip.split('.')[0]]
        elif args.seg_method=='pair':
            with open(args.data_path + 'pair_cps_all.p', 'rb') as f:
                check_points = pickle.load(f)
            cps_T=check_points[clip.split('.')[0]]
            cps_B=check_points[clip.split('.')[0]]
        elif args.seg_method=='combination1':
            # 'kernel_kmeans','dbscan55','hdbscan'
            with open(args.cps_path + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T = check_points_tracker['kernel_kmeans']
            cps_B = check_points_battery['kernel_kmeans']
            cps_T.extend(check_points_tracker['dbscan55'])
            cps_B.extend(check_points_battery['dbscan55'])
            with open(args.data_path + 'check_points_hdbscan/' + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T.extend(check_points_tracker['hdbscan'])
            cps_B.extend(check_points_battery['hdbscan'])
        elif args.seg_method=='combination2':
            # 'kernel_kmeans','dbscan55','hdbscan', 'pair'
            with open(args.cps_path + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T = check_points_tracker['kernel_kmeans']
            cps_B = check_points_battery['kernel_kmeans']
            cps_T.extend(check_points_tracker['dbscan55'])
            cps_B.extend(check_points_battery['dbscan55'])
            with open(args.data_path + 'check_points_hdbscan/' + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T.extend(check_points_tracker['hdbscan'])
            cps_B.extend(check_points_battery['hdbscan'])
            with open(args.data_path + 'pair_cps_all.p', 'rb') as f:
                check_points = pickle.load(f)
            cps_T.extend(check_points[clip.split('.')[0]])
            cps_B.extend(check_points[clip.split('.')[0]])
        elif args.seg_method=='combination3':
            #'kernel_kmeans','dbscan55','hdbscan','pair', 'dbscan23'
            with open(args.cps_path + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T = check_points_tracker['kernel_kmeans']
            cps_B = check_points_battery['kernel_kmeans']
            cps_T.extend(check_points_tracker['dbscan55'])
            cps_B.extend(check_points_battery['dbscan55'])
            cps_T.extend(check_points_tracker['dbscan23'])
            cps_B.extend(check_points_battery['dbscan23'])
            with open(args.data_path + 'check_points_hdbscan/' + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T.extend(check_points_tracker['hdbscan'])
            cps_B.extend(check_points_battery['hdbscan'])
            with open(args.data_path + 'pair_cps_all.p', 'rb') as f:
                check_points = pickle.load(f)
            cps_T.extend(check_points[clip.split('.')[0]])
            cps_B.extend(check_points[clip.split('.')[0]])
        elif args.seg_method=='combination4':
            # 'kernel_kmeans','dbscan55','hdbscan','pair', 'KTS'
            with open(args.cps_path + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T = check_points_tracker['kernel_kmeans']
            cps_B = check_points_battery['kernel_kmeans']
            cps_T.extend(check_points_tracker['dbscan55'])
            cps_B.extend(check_points_battery['dbscan55'])
            with open(args.data_path + 'check_points_hdbscan/' + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T.extend(check_points_tracker['hdbscan'])
            cps_B.extend(check_points_battery['hdbscan'])
            with open(args.data_path + 'pair_cps_all.p', 'rb') as f:
                check_points = pickle.load(f)
            cps_T.extend(check_points[clip.split('.')[0]])
            cps_B.extend(check_points[clip.split('.')[0]])
            with open(args.data_path + 'KTS_seg_points.p', 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T.extend(check_points_tracker[clip.split('.')[0]])
            cps_B.extend(check_points_battery[clip.split('.')[0]])
        else:
            with open(args.data_path+'check_points/' + clip, 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
            cps_T = check_points_tracker[args.seg_method]
            cps_B = check_points_battery[args.seg_method]

        cps_T = sorted(np.unique(np.array(cps_T)))
        if args.concate:
            cps_T = concate_check_points_less_than_N(cps_T, args)

        cps_B = sorted(np.unique(np.array(cps_B)))
        if args.concate:
            cps_B = concate_check_points_less_than_N(cps_B, args)

        cps_comb1_T[clip]=cps_T
        cps_comb1_B[clip]=cps_B
    with open(args.data_path2+ 'cps_'+args.seg_method+'.p', 'wb') as f:
       pickle.dump([cps_comb1_T, cps_comb1_B], f)


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

    with open(path.init_cps, 'rb') as f:
        cps_old=pickle.load(f)

    cps_T=cps_old[0]
    cps_B=cps_old[1]

    #for clip in clips_all:

    for clip in ['test1.p']:
        cps_T[clip]=[]
        cps_B[clip]=[]

        #for method in ['dbscan55', 'pair', 'hdbscan', 'kernel_kmeans', 'KTS']:
        for method in ['pair']:

            with open(args.data_path2+ 'cps_'+method+'.p', 'rb') as f:
                check_points_tracker, check_points_battery = pickle.load(f)
                print('pair {}'.format(check_points_tracker[clip]))
                print('pair {}'.format(check_points_battery[clip]))

                cps_T[clip].extend(check_points_tracker[clip])
                cps_B[clip].extend(check_points_battery[clip])

        cps_T[clip].extend(list(range(0, clips_len[clip], 60)))
        cps_B[clip].extend(list(range(0, clips_len[clip], 60)))
        cps_T[clip] = sorted(np.unique(np.array(cps_T[clip])))
        cps_T[clip] = concate_check_points_less_than_N(cps_T[clip], args)
        cps_B[clip] = sorted(np.unique(np.array(cps_B[clip])))
        cps_B[clip] = concate_check_points_less_than_N(cps_B[clip], args)
        # add more init segs, per 60 frames

        # if clip.split('.')[0] in event_seg_tracker:
        #     print(cps_T[clip])
        #     print(seg2cp(event_seg_tracker[clip.split('.')[0]]))
        #     print(cps_B[clip])
        #     print(seg2cp(event_seg_battery[clip.split('.')[0]]))


    with open(os.path.join(args.cps_path, 'CPS_NEW_0601.p'), 'wb') as f:
        pickle.dump([cps_T, cps_B], f)


