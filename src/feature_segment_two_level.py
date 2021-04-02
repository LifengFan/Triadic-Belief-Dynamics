import os
import glob
import pickle
from feature_embed import MLP
import torch
import numpy as np
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import copy
import metadata

class TwolevelSegment:
    def __init__(self):
        # load model
        net = MLP()
        net.load_state_dict(torch.load('./cptk/model_490.pth'))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        self.net = net

        # tracker
        self.trackers = metadata.tracker_skeID
        self.event_seg_tracker = metadata.event_seg_tracker
        self.event_seg_battery = metadata.event_seg_battery

    def check_end(self, seg, length):
        if seg[1] == length - 1:
            end = seg[1]
        else:
            end = seg[1] + 1
        return seg[0], end

    def id2labels(self, idx, length):
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

    def find_parent_id(self, seg_id, seg_mark):
        while seg_mark[seg_id]:
            seg_id = seg_mark[seg_id]
        return seg_id

    def concate_labels(self, labels):
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
                parent_id = self.find_parent_id(seg_id, seg_mark)

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

        ## added by Lifeng
        if (labels[0][1]-labels[0][0])<10:
            labels[1][0]=labels[0][0]
            del labels[0]

        return labels

    def sub_segment_single(self, second_level_seg, seg, features):
        cluster_num = 2
        start, end = self.check_end(seg, features.shape[0])
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10, max_iter=100).fit(features[start:end])
        sub_labels = self.id2labels(kmeans.labels_, end - start)
        # for seg_id, seg in enumerate(sub_labels):
        #     print(seg_id, seg[1] - seg[0])
        sub_labels = self.concate_labels(sub_labels)
        for sub_id, sub_label in enumerate(sub_labels):
            start_sub, end_sub = self.check_end(sub_label, end)
            second_level_seg[start_sub + start:end_sub + start] = sub_id
        return second_level_seg

    def sub_segment_pair(self, second_level_battery_seg, second_level_tracker_seg, seg, features):
        cluster_num = 2
        start, end = self.check_end(seg, features.shape[0])
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10, max_iter=100).fit(features[start:end])
        sub_labels = self.id2labels(kmeans.labels_, end - start)
        sub_labels = self.concate_labels(sub_labels)

        for sub_id, sub_label in enumerate(sub_labels):
            start_sub, end_sub = self.check_end(sub_label, end)
            second_level_battery_seg[start_sub + start:end_sub + start] = sub_id
            second_level_tracker_seg[start_sub + start:end_sub + start] = sub_id
        return second_level_battery_seg, second_level_tracker_seg

    def vis_seg(self, img_files, first_level_seg, second_level_tracker_seg, second_level_battery_seg, clip):
        save_path = './seg_vis/two_level/'
        save_img_path = './seg_vis/two_level/' + clip + '/'
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        for i in range(1, 3):
            if str(i) == self.trackers[clip].split('.')[0][-1]:
                p_type = 'tracker'
                second_level_seg = second_level_tracker_seg
            else:
                p_type = 'battery'
                second_level_seg = second_level_battery_seg

            filename1 = save_path + clip + '_' + p_type + '.avi'
            video_shape = (800, 480)
            out = cv2.VideoWriter(filename1, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 24, video_shape)
            for frame_id, img_name in enumerate(img_files):
                img = cv2.imread(img_name)
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                img = cv2.putText(img, '{}_{}:first_level_label:{}'.format(str(frame_id),p_type, first_level_seg[frame_id]), org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                img = cv2.putText(img, '{}_{}:second_level_label:{}'.format(str(frame_id),p_type, second_level_seg[frame_id]), (org[0], org[1]+50), font,
                                  fontScale, color, thickness, cv2.LINE_AA)
                # cv2.imwrite(save_img_path + p_type + '_{0:04}.jpg'.format(frame_id), img)
                out.write(cv2.resize(img, video_shape))
            out.release()

    def plot_segmentation(self, input_labels_list, input_labels_list_gt, endframe, first_level_seg, first_level_seg_gt,
                          vmax=None, filename=None, border=True,
                          cmap=plt.get_cmap('gist_rainbow')):
        plt_idx = 0
        aspect_ratio = 30
        fig = plt.figure(figsize=(28, 5))
        first_level_seg_image = np.empty((int(endframe / aspect_ratio), endframe, 3))
        colors = [[255, 0, 0], [0, 255, 0],
                  [0, 0, 255], [255, 255, 0], [0, 255, 255],
                  [255, 0, 255],[255, 100, 255], [66, 245, 66],
                  [245, 66, 221], [66, 245, 236], [245, 66, 78],
                  [245, 164, 66], [66, 245, 170], [245, 206, 66],
                  [132, 66, 245], [66, 87, 245], [191, 245, 66],
                  [255, 0, 0], [0, 255, 0],
                  [0, 0, 255], [255, 255, 0], [0, 255, 255],
                  [255, 0, 255], [255, 100, 255], [66, 245, 66],
                  [245, 66, 221], [66, 245, 236], [245, 66, 78],
                  [245, 164, 66], [66, 245, 170], [245, 206, 66],
                  [132, 66, 245], [66, 87, 245], [191, 245, 66],
                  [255, 0, 0], [0, 255, 0],
                  [0, 0, 255], [255, 255, 0], [0, 255, 255],
                  [255, 0, 255], [255, 100, 255], [66, 245, 66],
                  [245, 66, 221], [66, 245, 236], [245, 66, 78],
                  [245, 164, 66], [66, 245, 170], [245, 206, 66],
                  [132, 66, 245], [66, 87, 245], [191, 245, 66]
                  ]
        for frame in range(endframe):
            # first_level_seg_image[:, frame] = int(first_level_seg[frame]) + 1
            first_level_seg_image[:, frame, 0] = colors[int(first_level_seg[frame]) + 1][0]
            first_level_seg_image[:, frame, 1] = colors[int(first_level_seg[frame]) + 1][1]
            first_level_seg_image[:, frame, 2] = colors[int(first_level_seg[frame]) + 1][2]

        first_level_seg_image_gt = np.empty((int(endframe / aspect_ratio), endframe, 3))

        for frame in range(endframe):
            # first_level_seg_image_gt[:, frame] = int(first_level_seg_gt[frame]) + 1
            first_level_seg_image_gt[:, frame, 0] = colors[int(first_level_seg_gt[frame]) + 1][0]
            first_level_seg_image_gt[:, frame, 1] = colors[int(first_level_seg_gt[frame]) + 1][1]
            first_level_seg_image_gt[:, frame, 2] = colors[int(first_level_seg_gt[frame]) + 1][2]

        # inserted_idx = []
        # for i in np.arange(endframe - 2, -1, -1):
        #     if first_level_seg[i] != first_level_seg[i+1]:
        #         inserted_idx.append(i + 1)
        #         first_level_seg_image = np.insert(first_level_seg_image, i + 1,
        #                                           -np.zeros((1, (int(endframe / aspect_ratio)))), axis = 1)
        #
        # inserted_idx_gt = []
        # for i in np.arange(endframe - 2, -1, -1):
        #     if first_level_seg_gt[i] != first_level_seg_gt[i + 1]:
        #         inserted_idx_gt.append(i + 1)
        #         first_level_seg_image_gt = np.insert(first_level_seg_image_gt, i + 1,
        #                                           np.zeros((1, (int(endframe / aspect_ratio)))), axis=1)
        first_segs = [first_level_seg_image, first_level_seg_image_gt, first_level_seg_image, first_level_seg_image_gt]
        second_segs = [input_labels_list[0], input_labels_list_gt[0], input_labels_list[1], input_labels_list_gt[1]]
        # print(np.unique(input_labels_list[0]))
        # print(np.unique(input_labels_list_gt[0]))
        # print(np.unique(input_labels_list[1]))
        # print(np.unique(input_labels_list_gt[1]))
        # cmaps = colors.ListedColormap(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
        for pid in range(0, 4):
            seg_image = np.empty((int(endframe / aspect_ratio), endframe, 3))
            input_labels = second_segs[pid]
            for frame in range(endframe):
                # seg_image[:, frame] = input_labels[frame] + 3
                seg_image[:, frame, 0] = colors[int(input_labels[frame]) + 3][0]
                seg_image[:, frame, 1] = colors[int(input_labels[frame]) + 3][1]
                seg_image[:, frame, 2] = colors[int(input_labels[frame]) + 3][2]
                # for idx in inserted_idx:
                #     seg_image = np.insert(seg_image, idx, np.zeros((1, (int(endframe / aspect_ratio)))), axis = 1)
            total_seg_image = np.vstack([first_segs[pid], seg_image])
            plt_idx += 1
            ax = plt.subplot(4, 1, plt_idx)
            if pid == 0:
                ax.set_ylabel('battery')
            elif pid == 1:
                ax.set_ylabel('battery_gt')
            elif pid == 2:
                ax.set_ylabel('tracker')
            else:
                ax.set_ylabel('tracker_gt')
            if not border:
                ax.axis('off')
            # if vmax:
            #     ax.imshow(total_seg_image, vmin=0, vmax=vmax, cmap=cmap)
            # else:
            #     ax.imshow(total_seg_image, cmap=cmap)
            ax.imshow(total_seg_image)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def segment(self, img_names, pair_features, single_features_battery, single_features_tracker, clip):
        total_labels = {}
        # joint or individual
        input = torch.from_numpy(pair_features).float().cuda()
        predicted_val, embedding = self.net(torch.autograd.Variable(input))
        predicted_val = predicted_val.data
        max_score, idx = torch.max(predicted_val, 1)
        idx = idx.cpu().numpy()
        labels = self.id2labels(idx, pair_features.shape[0])

        first_level_seg = np.zeros(len(img_names))
        for seg_id, seg in enumerate(labels):
            start, end = self.check_end(seg, len(img_names))
            first_level_seg[start:end] = seg[2]

        total_labels['level1'] = first_level_seg

        second_level_tracker_seg = np.zeros(len(img_names))
        second_level_battery_seg = np.zeros(len(img_names))
        for seg_id, seg in enumerate(labels):
            # length < 10
            if seg[1] - seg[0] < 10:
                start, end = self.check_end(seg, len(img_names))
                second_level_tracker_seg[start:end] = 0
                second_level_battery_seg[start:end] = 0
                continue

            # length > 10
            if seg[2] == 0:
                second_level_battery_seg = self.sub_segment_single(second_level_battery_seg, seg, single_features_battery)
                second_level_tracker_seg = self.sub_segment_single(second_level_tracker_seg, seg, single_features_tracker)
            else:
                second_level_battery_seg, second_level_tracker_seg = self.sub_segment_pair(second_level_battery_seg,
                                                                                      second_level_tracker_seg, seg, pair_features)
        total_labels['level2_tracker'] = second_level_tracker_seg
        total_labels['level2_battery'] = second_level_battery_seg

        # vis
        # self.vis_seg(img_names, first_level_seg, second_level_tracker_seg, second_level_battery_seg, clip)
        gt = self.event_seg_tracker[clip]
        second_level_tracker_seg_gt = np.zeros(len(img_names))
        for seg_id, seg in enumerate(gt):
            if seg[1] == len(img_names) - 1:
                end = seg[1]
            else:
                end = seg[1] + 1
            second_level_tracker_seg_gt[seg[0]:end] = seg_id

        gt = self.event_seg_battery[clip]
        second_level_battery_seg_gt = np.zeros(len(img_names))
        for seg_id, seg in enumerate(gt):
            if seg[1] == len(img_names) - 1:
                end = seg[1]
            else:
                end = seg[1] + 1
            second_level_battery_seg_gt[seg[0]:end] = seg_id

        gt = self.event_seg_tracker[clip]
        first_level_seg_gt = np.zeros(len(img_names))
        for seg_id, seg in enumerate(gt):
            if seg[1] == len(img_names) - 1:
                end = seg[1]
            else:
                end = seg[1] + 1
            if seg[2] == 0:
                first_level_seg_gt[seg[0]:end] = 0
            else:
                first_level_seg_gt[seg[0]:end] = 1

        first_level_gt_labels = self.id2labels(first_level_seg_gt, first_level_seg_gt.shape[0])
        gt = self.event_seg_tracker[clip]
        second_level_tracker_seg_gt = np.zeros(len(img_names))
        count = 0
        sub_count = 0
        for seg_id, seg in enumerate(gt):
            if seg[1] <= first_level_gt_labels[count][1]:
                if seg[1] == len(img_names) - 1:
                    end = seg[1]
                else:
                    end = seg[1] + 1
                second_level_tracker_seg_gt[seg[0]:end] = sub_count
                sub_count += 1
            else:
                count += 1
                sub_count = 0
                if seg[1] == len(img_names) - 1:
                    end = seg[1]
                else:
                    end = seg[1] + 1
                second_level_tracker_seg_gt[seg[0]:end] = sub_count
                sub_count += 1

        gt = self.event_seg_battery[clip]
        second_level_battery_seg_gt = np.zeros(len(img_names))
        count = 0
        sub_count = 0
        for seg_id, seg in enumerate(gt):
            if seg[1] <= first_level_gt_labels[count][1]:
                if seg[1] == len(img_names) - 1:
                    end = seg[1]
                else:
                    end = seg[1] + 1
                second_level_battery_seg_gt[seg[0]:end] = sub_count
                sub_count += 1
            else:
                count += 1
                sub_count = 0
                if seg[1] == len(img_names) - 1:
                    end = seg[1]
                else:
                    end = seg[1] + 1
                second_level_battery_seg_gt[seg[0]:end] = sub_count
                sub_count += 1



        self.plot_segmentation([second_level_battery_seg, second_level_tracker_seg], [second_level_battery_seg_gt, second_level_tracker_seg_gt],
                               second_level_tracker_seg.shape[0], first_level_seg, first_level_seg_gt)

        return first_level_seg, second_level_tracker_seg, second_level_tracker_seg

    def load_file(self):
        # data path
        pair_feature_path = '/home/shuwen/data/data_preprocessing2/feature_pair/'
        single_feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
        img_path = '/home/shuwen/data/data_preprocessing2/annotations/'

        # segment
        # clips = os.listdir(img_path)
        clips = self.event_seg_battery.keys()
        # clips = ['test_94342_16']
        for clip in clips[:5]:
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
                if str(i) == self.trackers[clip].split('.')[0][-1]:
                    single_features_tracker = single_features[i]
                else:
                    single_features_battery = single_features[i]
            self.segment(img_names, pair_features, single_features_battery, single_features_tracker, clip)

if __name__ == '__main__':
    segmenter = TwolevelSegment()
    segmenter.load_file()