import pickle
import argparse
import metadata
import hdbscan
from metric import *
import numpy as np
import time
import os
import glob
import cv2


class Hdbscan_Clusterer:

    def __init__(self):

        with open('../data/event_seg_tracker', 'rb') as f:
             self.event_seg_tracker=pickle.load(f)

        with open('../data/event_seg_battery', 'rb') as f:
             self.event_seg_battery=pickle.load(f)

    def cluster(self, feature_file, min_cluster_size):

        with open(feature_file, 'rb') as f:
            features = pickle.load(f)

        p1_feature = features[1]
        p2_feature = features[2]
        features = np.vstack([p1_feature, p2_feature])

        hdbscan_cluster1 = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(p1_feature)
        hdbscan_cluster2 = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(p2_feature)

        labels1 = []
        labels1.append([0])
        counter = 0
        for i in range(1, p1_feature.shape[0]):
            if hdbscan_cluster1[i] != hdbscan_cluster1[i - 1]:
                labels1[counter].extend([i - 1, hdbscan_cluster1[i - 1]])
                labels1.append([i])
                counter += 1
        if len(labels1[counter]) < 2:
            labels1[counter].extend([p1_feature.shape[0] - 1, hdbscan_cluster1[p1_feature.shape[0] - 1]])

        # print feature_file.split('/')[-1].split('.')[0]

        if metadata.trackers_type[feature_file.split('/')[-1].split('.')[0]] == 'skele1.p':
            gt1 = self.event_seg_tracker[feature_file.split('/')[-1].split('.')[0]]
            type1='tracker'
        elif metadata.trackers_type[feature_file.split('/')[-1].split('.')[0]] == 'skele2.p':
            gt1 = self.event_seg_battery[feature_file.split('/')[-1].split('.')[0]]
            type1='battery'

        error1, err1_L, err1_N = segment_error(labels1, gt1, args.seg_err_alpha)

        labels2 = []
        labels2.append([0])
        counter = 0
        for i in range(1, p2_feature.shape[0]):
            if hdbscan_cluster2[i] != hdbscan_cluster2[i - 1]:
                labels2[counter].extend([i - 1, hdbscan_cluster2[i - 1]])
                labels2.append([i])
                counter += 1
        if len(labels2[counter]) < 2:
            labels2[counter].extend([p2_feature.shape[0] - 1, hdbscan_cluster2[p2_feature.shape[0] - 1]])

        if metadata.trackers_type[feature_file.split('/')[-1].split('.')[0]] == 'skele1.p':
            gt2 = self.event_seg_battery[feature_file.split('/')[-1].split('.')[0]]
            type2='battery'
        elif metadata.trackers_type[feature_file.split('/')[-1].split('.')[0]] == 'skele2.p':
            gt2 = self.event_seg_tracker[feature_file.split('/')[-1].split('.')[0]]
            type2='tracker'

        error2, err2_L, err2_N = segment_error(labels2, gt2, args.seg_err_alpha)

        print labels1, labels2

        return error1 + error2, err1_L+err2_L, err1_N+err2_N, labels1, gt1, type1, labels2, gt2, type2


def run_clustering(args):

    clusterer = Hdbscan_Clusterer()
    clips = clusterer.event_seg_tracker.keys()

    errors = []
    min_error = 10000000
    min_para = None

    start=time.time()

    for min_cluster_size in range(100,300,10):
        total_error = []
        total_err_L=[]
        total_err_N=[]

        for clip in clips:
            error, err_L, err_N, _ , _, _, _, _, _= clusterer.cluster(args.data_path + clip + '.p', min_cluster_size)
            total_error.append(error)
            total_err_L.append(err_L)
            total_err_N.append(err_N)
            #print args.seg_err_alpha, clip, err_L, err_N

        mean_error = np.array(total_error).mean()
        mean_err_L=np.array(total_err_L).mean()
        mean_err_N=np.array(total_err_N).mean()

        print('mean err L:{}, mean err N:{}'.format(mean_err_L, mean_err_N))

        errors.append([mean_error])

        if mean_error < min_error:
            min_error = mean_error
            min_para = min_cluster_size

        end=time.time()

        print('time cost: {}s, alpha:{}, min-cluster-size:{}, mean-error:{}, min-error:{}, min-para:{}'.format(end-start,
                args.seg_err_alpha, min_cluster_size, mean_error, min_error, min_para))


def test_segments(feature_file, img_files):

        save_path = './seg_vis/dbscan/'+feature_file.split('/')[-1]
        # print feature_file.split('/')[-1][:-2]
        # print "save_path: {}".format(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        clusterer = Hdbscan_Clusterer()

        _, _, _, labels1, gt1, type1, labels2, gt2, type2 = clusterer.cluster(feature_file, args.hdbscan_min_cluster_size)

        hdbscan_seg1 = np.zeros(len(img_files))
        for seg_id, seg in enumerate(labels1):
            if seg[1] == len(img_files) - 1:
                end = seg[1]
            else:
                end = seg[1] + 1
            hdbscan_seg1[seg[0]:end] = seg_id

        gt_seg1 = np.zeros(len(img_files))
        for seg_id, seg in enumerate(gt1):
            if seg[1] == len(img_files) - 1:
                end = seg[1]
            else:
                end = seg[1] + 1
            gt_seg1[seg[0]:end] = seg_id

        hdbscan_seg2 = np.zeros(len(img_files))
        for seg_id, seg in enumerate(labels2):
            if seg[1] == len(img_files) - 1:
                end = seg[1]
            else:
                end = seg[1] + 1
            hdbscan_seg2[seg[0]:end] = seg_id

        gt_seg2 = np.zeros(len(img_files))
        for seg_id, seg in enumerate(gt2):
            if seg[1] == len(img_files) - 1:
                end = seg[1]
            else:
                end = seg[1] + 1
            gt_seg2[seg[0]:end] = seg_id

        for frame_id, img_name in enumerate(img_files):
            img = cv2.imread(img_name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            img = cv2.putText(img, '{}_gt:{}'.format(type1, gt_seg1[frame_id]), org, font,
                              fontScale, color, thickness, cv2.LINE_AA)
            img = cv2.putText(img, '{}_hdbscan:{}'.format(type1, hdbscan_seg1[frame_id]), (org[0], org[1] + 50), font,
                              fontScale, color, thickness, cv2.LINE_AA)
            img = cv2.putText(img, '{}_gt:{}'.format(type2, gt_seg2[frame_id]), (org[0], org[1] + 100), font,
                              fontScale, color, thickness, cv2.LINE_AA)
            img = cv2.putText(img, '{}_hdbscan:{}'.format(type2, hdbscan_seg2[frame_id]), (org[0], org[1] + 150), font,
                              fontScale, color, thickness, cv2.LINE_AA)

            cv2.imwrite(save_path  + '/_{0:04}.jpg'.format(frame_id), img)


def parse_arguments():

    parser=argparse.ArgumentParser(description='HDBSCAN Clustering')

    parser.add_argument('--data-path', default='/home/lfan/Dropbox/Projects/ECCV20/data/feature_gaze_360_hog/')
    parser.add_argument('--seg-err-alpha',default = 12, help='project root path')
    parser.add_argument('--hdbscan-min-cluster-size', default=210)

    return parser.parse_args()

if __name__=='__main__':

    args=parse_arguments()
    #run_clustering(args)
    # test_segments(args.data_path+'/test_94342_16.p', img_names)

    files=sorted(glob.glob(args.data_path+'*.p'))
    for i, filename in enumerate(files):
        print filename
        img_names = sorted(glob.glob('/home/lfan/Dropbox/Projects/ECCV20/annotations/'+filename.split('/')[-1].split('.')[0]+'/kinect/*.jpg'))
        test_segments(filename, img_names)


