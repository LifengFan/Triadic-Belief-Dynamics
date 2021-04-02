import os
import glob
import pickle
import  sys
sys.path.append('/home/shuwen/data/Six-Minds-Project/data_processing_scripts/')
from metadata import *
import torch
from utils import *
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
import hdbscan
import matplotlib.pyplot as plt
import argparse

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


def sub_segment_pair(second_level_battery_seg, second_level_tracker_seg, seg, features):
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
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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

            # sub_check_points_kmeans, sub_check_points_kernel_kmeans, sub_check_points_agg, sub_check_points_dbscan32, \
            # sub_check_points_dbscan23, sub_check_points_dbscan55 = sub_segment_single(second_level_battery_seg, seg,
            #                                                                single_features_battery)
            # check_points_battery['k-means'].extend(sub_check_points_kmeans)
            # check_points_battery['kernel_kmeans'].extend(sub_check_points_kernel_kmeans)
            # check_points_battery['agg'].extend(sub_check_points_agg)
            # check_points_battery['dbscan32'].extend(sub_check_points_dbscan32)
            # check_points_battery['dbscan23'].extend(sub_check_points_dbscan23)
            # check_points_battery['dbscan55'].extend(sub_check_points_dbscan55)
            #
            # sub_check_points_kmeans, sub_check_points_kernel_kmeans, sub_check_points_agg, sub_check_points_dbscan32, \
            # sub_check_points_dbscan23, sub_check_points_dbscan55 = sub_segment_single(second_level_tracker_seg, seg,
            #                                                                single_features_tracker)
            # check_points_tracker['k-means'].extend(sub_check_points_kmeans)
            # check_points_tracker['kernel_kmeans'].extend(sub_check_points_kernel_kmeans)
            # check_points_tracker['agg'].extend(sub_check_points_agg)
            # check_points_tracker['dbscan32'].extend(sub_check_points_dbscan32)
            # check_points_tracker['dbscan23'].extend(sub_check_points_dbscan23)
            # check_points_tracker['dbscan55'].extend(sub_check_points_dbscan55)
            sub_check_points_hdbsan = sub_segment_single(second_level_battery_seg, seg, single_features_battery)
            check_points_battery['hdbscan'].extend(sub_check_points_hdbsan)
            sub_check_points_hdbsan = sub_segment_single(second_level_tracker_seg, seg, single_features_tracker)
            check_points_tracker['hdbscan'].extend(sub_check_points_hdbsan)

        else:
            sub_check_points_hdbsan = sub_segment_pair(second_level_battery_seg, second_level_tracker_seg, seg, pair_features)
            check_points_battery['hdbscan'].extend(sub_check_points_hdbsan)

    return check_points_tracker, check_points_battery

def generate_check_point(args):
    # data path
    pair_feature_path = args.data_path+'feature_pair/'
    single_feature_path = args.data_path+'feature_single/'
    img_path = args.project_path+'/annotations/'
    save_path=args.save_path+'/seg_points/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # segment
    clips = os.listdir(img_path)

    # load model
    net = MLP()
    net.load_state_dict(torch.load('./model_490.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    #pair_cps_all={}

    for clip in clips:
        img_names = sorted(glob.glob(os.path.join(img_path, clip, 'kinect/*.jpg')))
        if not os.path.exists(os.path.join(pair_feature_path, clip + '.p')):
            continue
        if not os.path.exists(os.path.join(single_feature_path, clip + '.p')):
            continue
        print(clip)
        with open(os.path.join(pair_feature_path, clip + '.p'), 'rb') as f:
            pair_features = pickle.load(f)
        with open(os.path.join(single_feature_path, clip + '.p'), 'rb') as f:
            single_features = pickle.load(f)

        for i in range(1, 3):
            if str(i) == tracker_skeID[clip].split('.')[0][-1]:
                single_features_tracker = single_features[i]
            else:
                single_features_battery = single_features[i]

        check_points_tracker,check_points_battery = segment(img_names, pair_features, single_features_battery, single_features_tracker,clip, net, args)
        #pair_cps= segment(img_names, pair_features, single_features_battery,single_features_tracker, clip, net, args)
        #pair_cps_all[clip]=pair_cps

        #with open(save_path+ '/pair_cps_all.p', 'wb') as f:
        #    pickle.dump(pair_cps_all, f)

def seg_vis(cps, X, gt, name):
    plt.figure("")
    plt.plot(X)
    print "Estimated:", cps
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
      plt.plot([cp, cp], [mi, ma], 'r')
    for cp in gt:
      plt.plot([cp, cp], [mi, ma], 'g')
    plt.title(name)
    plt.show()
    print "="*79

def concate_check_points_less_than_10(check_points):
    new_check_points = []
    new_check_points.append(check_points[0])
    cnt = 0
    for i in range(1, len(check_points)):
        if check_points[i] - new_check_points[cnt] < 50:
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

    with open(args.data_path+"/HACA_seg_points.p", 'rb') as f:
        haca_tracker, haca_battery = pickle.load(f)
    with open(args.data_path+"/KTS_seg_points.p", 'rb') as f:
        kts_tracker, kts_battery = pickle.load(f)
    with open(args.data_path+"pair_cps_all.p", 'rb') as f:
        pair_cps_all = pickle.load(f)

    for clip in clips[:10]:
        with open(os.path.join(single_feature_path, clip), 'rb') as f:
            single_features = pickle.load(f)
        if clip.split('.')[0] not in event_seg_tracker:
            continue
        tracker_gt_seg = event_seg_tracker[clip.split('.')[0]]
        tracker_gt_cp = seg2cp(tracker_gt_seg)
        battery_gt_seg = event_seg_battery[clip.split('.')[0]]
        battery_gt_cp = seg2cp(battery_gt_seg)
        for i in range(1, 3):
            if str(i) == tracker_skeID[clip.split('.')[0]].split('.')[0][-1]:
                single_features_tracker = single_features[i]
            else:
                single_features_battery = single_features[i]
        with open(check_point_path + clip, 'rb') as f:
            check_points_tracker, check_points_battery = pickle.load(f)
        # check_points_total_tracker = []
        # check_points_total_battery = []
        # # for cluster in check_points_tracker.keys()[5:]:
        # #     check_points_total_tracker.extend(check_points_tracker[cluster])
        # #     check_points_total_battery.extend(check_points_battery[cluster])
        # # with open('./check_points_hdbscan/' + clip, 'rb') as f:
        # #     check_points_tracker, check_points_battery = pickle.load(f)
        # # for cluster in check_pointsseg_vis(cps, X, gt, name)_tracker.keys():
        # #     check_points_total_tracker.extend(check_points_tracker[cluster])
        # #     check_points_total_battery.extend(check_points_battery[cluster])
        # check_points_total_tracker.extend(kts_tracker[clip.split('.')[0]])
        # check_points_total_battery.extend(kts_battery[clip.split('.')[0]])
        # check_points_total_tracker = np.array(check_points_total_tracker)
        # # check_points_total_tracker = np.append(check_points_total_tracker, haca_tracker[clip]['s_top'])
        # check_points_unique_tracker = np.unique(check_points_total_tracker)
        # check_points_unique_tracker = sorted(check_points_unique_tracker)
        # # check_points_unique_tracker = concate_check_points_less_than_10(check_points_unique_tracker)
        # seg_vis(check_points_unique_tracker, single_features_tracker, tracker_gt_cp, clip.split('.')[0] + ':kts:tracker')
        # check_points_total_battery = np.array(check_points_total_battery)
        # # check_points_total_battery = np.append(check_points_total_battery, haca_battery[clip]['s_top'])
        # check_points_unique_battery = np.unique(check_points_total_battery)
        # check_points_unique_battery = sorted(check_points_unique_battery)
        # # check_points_unique_battery = concate_check_points_less_than_10(check_points_unique_battery)
        # seg_vis(check_points_unique_battery, single_features_battery, battery_gt_cp, clip.split('.')[0] + ':kts:battery')

        seg_vis(pair_cps_all[clip.split('.')[0]], single_features_tracker, tracker_gt_cp, clip.split('.')[0] + ':pair:tracker')
        #seg_vis(pair_cps_all[clip.split('.')[0]], single_features_battery, battery_gt_cp, clip.split('.')[0] + ':pair:battery')

def parse_arguments():

    parser=argparse.ArgumentParser(description='generate_seg_points')
    parser.add_argument('--project-path',default = '/home/lfan/Dropbox/Projects/NIPS20/')
    parser.add_argument('--data-path', default='/home/lfan/Dropbox/Projects/NIPS20/data/')
    parser.add_argument('--save-path', default='/home/lfan/Dropbox/Projects/NIPS20/result/')
    parser.add_argument('--seg-method')
    parser.add_argument('--cps-path', default='/home/lfan/Dropbox/Projects/NIPS20/data/check_points/')
    parser.add_argument('--single-feature-path', default='/home/lfan/Dropbox/Projects/NIPS20/data/feature_single/')
    parser.add_argument('--clips', default=os.listdir('/home/lfan/Dropbox/Projects/NIPS20/data/check_points/'))

    return parser.parse_args()



if __name__ == '__main__':

    args=parse_arguments()
    #generate_check_point(args)
    #merge_check_point()
    #---------------------------------------
    args.save_path=args.save_path+''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    select_cps(args)

