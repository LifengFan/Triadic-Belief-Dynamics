import os
import glob
import torch
from torchvision import transforms
import cv2
import joblib
import pickle
import numpy as np
from beam_search_m import *
from scipy import sparse


class mydataset_atomic_first_view(torch.utils.data.Dataset):
    def __init__(self, seq):
        self.seq = seq
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
        self.tracker_head = './tracker_head_patch/'
        self.battery_head = './battery_head_patch/'
        self.obj_patch = './obj_patch/'

    def __getitem__(self, index):
        rec = self.seq[index]
        head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224)) # [5,4,3,224,224]
        pos_sq = np.zeros((self.seq_size, self.node_num, 6)) #[5,4,6]
        hog_sq = np.zeros((self.seq_size, self.feature_dim*2))
        attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num)) #[5,4,4]
        clip, start_id, obj_list_seq, obj_list = rec

        with open(self.feature_path + clip, 'rb') as f:
            features = pickle.load(f, encoding='latin1')

        for sq_id in range(len(obj_list_seq)):
            fid = start_id + sq_id
            for node_i in [0, 1]:
                if node_i == 0:
                    # if fid not in tracker_bbox:
                    #     continue
                    head_box = np.load(os.path.join(self.tracker_head, clip, str(fid) + 'head' + '.npy'))
                    tracker_hog = features[1][fid][-self.feature_dim-10:-10]
                    pos_vec = np.load(os.path.join(self.tracker_head, clip, str(fid) + 'head_vec' + '.npy'))
                else:
                    # if fid not in battery_bbox:
                    #     continue
                    head_box = np.load(os.path.join(self.battery_head, clip, str(fid) + 'head' + '.npy'))
                    battery_hog = features[2][fid][-self.feature_dim-10:-10]
                    pos_vec = np.load(os.path.join(self.battery_head, clip, str(fid) + 'head_vec' + '.npy'))

                head_patch_sq[sq_id, node_i, ...] = head_box
                pos_sq[sq_id, node_i, :] = pos_vec
            hog_sq[sq_id, :] = np.hstack([tracker_hog, battery_hog])
            obj_id1, obj_id2, type1, type2, has_obj1, has_obj2 = obj_list_seq[sq_id]
            obj_name = obj_list[obj_id1]
            head = np.load(os.path.join(self.obj_patch, clip, str(obj_name) + '_' + str(fid) + '.npy'))
            head_patch_sq[sq_id, 2, ...] = head
            pos_vec = np.load(os.path.join(self.obj_patch, clip, str(obj_name) + '_' + str(fid) + 'vec.npy'))
            pos_sq[sq_id, 2, :] = pos_vec

            obj_name = obj_list[obj_id2]
            head = np.load(os.path.join(self.obj_patch, clip, str(obj_name) + '_' + str(fid) + '.npy'))
            head_patch_sq[sq_id, 2, ...] = head
            pos_vec = np.load(os.path.join(self.obj_patch, clip, str(obj_name) + '_' + str(fid) + 'vec.npy'))
            pos_sq[sq_id, 3, :] = pos_vec

            if has_obj1:
                if type1 == 'person':
                    attmat_sq[sq_id, 0, 1] = 1
                else:
                    attmat_sq[sq_id, 0, 2] = 1
            if has_obj2:
                if type2 == 'person':
                    attmat_sq[sq_id, 1, 0] = 1
                else:
                    attmat_sq[sq_id, 1, 3] = 1

        return head_patch_sq, pos_sq, attmat_sq, index, hog_sq

    def __len__(self):
        return len(self.seq)



def reformat_data(args):
    data_path = args.img_path
    person_tracker_bbox =args.tracker_bbox # '../../3d_pose2gaze/tracker_record_bbox/'
    person_battery_bbox = args.battery_bbox #'../../3d_pose2gaze/record_bbox/'
    attmat_path = args.attmat_path #'/home/shuwen/data/data_preprocessing2/record_attention_matrix/'
    obj_bbox_path = args.obj_bbox # '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
    tracker_head_save_path = args.data_path2+ 'tracker_head_patch/'
    battery_head_save_path = args.data_path2+'battery_head_patch/'
    obj_save_path = args.data_path2+'obj_patch/'
    if not os.path.exists(tracker_head_save_path):
        os.makedirs(tracker_head_save_path)
    if not os.path.exists(battery_head_save_path):
        os.makedirs(battery_head_save_path)
    if not os.path.exists(obj_save_path):
        os.makedirs(obj_save_path)


    clips = os.listdir(data_path)
    transformst = transforms.Compose([transforms.ToTensor()])
    for clip in clips:
        print(clip)
        tracker_save_prefix = os.path.join(tracker_head_save_path, clip)
        battery_save_prefix = os.path.join(battery_head_save_path, clip)
        obj_save_prefix = os.path.join(obj_save_path, clip)
        if not os.path.exists(tracker_save_prefix):
            os.makedirs(tracker_save_prefix)
        if not os.path.exists(battery_save_prefix):
            os.makedirs(battery_save_prefix)
        if not os.path.exists(obj_save_prefix):
            os.makedirs(obj_save_prefix)
        with open(person_tracker_bbox + clip + '.p', 'rb') as f:
            tracker_bbox = pickle.load(f, encoding='latin1')
        with open(person_battery_bbox + clip + '.p', 'rb') as f:
            battery_bbox = pickle.load(f, encoding='latin1')
        with open(attmat_path + clip + '.p', 'rb') as f:
            attmat, obj_list = pickle.load(f, encoding='latin1')
        img_names = sorted(glob.glob(os.path.join(data_path, clip, 'kinect/*.jpg')))
        assert len(img_names) > 0
        for fid, img_name in enumerate(img_names):
            img = cv2.imread(img_name)
            if fid in tracker_bbox and fid in battery_bbox:
                head_box = tracker_bbox[fid][0][0]
                img_patch = img[int(head_box[1]):int(head_box[3]), int(head_box[0]): int(head_box[2])]

                # img_patch = cv2.resize(img_patch, (224, 224))
                # head = transformst(img_patch).numpy()
                # for c in [0, 1, 2]:
                #     head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
                #np.save(os.path.join(tracker_save_prefix, str(fid) + 'head' + '.npy'), head)

                cv2.imwrite(os.path.join(tracker_save_prefix, str(fid) + 'head' + '.jpg'), img_patch)

                pos_vec = np.array([head_box[0] / img.shape[1], head_box[1] / img.shape[0], head_box[2] / img.shape[1],
                                    head_box[3] / img.shape[0], (head_box[0] + head_box[2]) / 2 / img.shape[1],
                                    (head_box[1] + head_box[3]) / 2 / img.shape[0]])
                np.save(os.path.join(tracker_save_prefix, str(fid) + 'head_vec' + '.npy'), pos_vec)

                head_box = battery_bbox[fid][0][0]
                img_patch = img[int(head_box[1]):int(head_box[3]), int(head_box[0]): int(head_box[2])]
                cv2.imwrite(os.path.join(battery_save_prefix, str(fid) + 'head' + '.jpg'), img_patch)

                pos_vec = np.array([head_box[0] / img.shape[1], head_box[1] / img.shape[0], head_box[2] / img.shape[1],
                                    head_box[3] / img.shape[0], (head_box[0] + head_box[2]) / 2 / img.shape[1],
                                    (head_box[1] + head_box[3]) / 2 / img.shape[0]])
                np.save(os.path.join(battery_save_prefix, str(fid) + 'head_vec' + '.npy'), pos_vec)

                for obj_name in obj_list:
                    with open(obj_bbox_path + clip.split('.')[0] + '/' + str(obj_name) + '.p', 'rb') as f:
                        obj_bbox = pickle.load(f, encoding='latin1')
                    obj_curr = obj_bbox[fid]
                    left = max(0, obj_curr[0])
                    top = max(0, obj_curr[1])
                    right = min(obj_curr[0] + obj_curr[2], img.shape[1])
                    bottom = min(obj_curr[1] + obj_curr[3], img.shape[0])
                    img_patch = img[int(top):int(bottom), int(left): int(right)]
                    if img_patch.shape[0] == 0 or img_patch.shape[1] == 0:
                        continue
                    cv2.imwrite(os.path.join(obj_save_prefix, str(obj_name) + '_' + str(fid) + '.jpg'), img_patch)
                    pos_vec = np.array(
                        [obj_curr[0] / img.shape[1], obj_curr[1] / img.shape[0], (obj_curr[2] + obj_curr[0]) / img.shape[1],
                         (obj_curr[3] + obj_curr[1]) / img.shape[0], (obj_curr[0] + obj_curr[2] / 2) / img.shape[1],
                         (obj_curr[1] + obj_curr[3] / 2) / img.shape[0]])
                    np.save(os.path.join(obj_save_prefix, str(obj_name) + '_' + str(fid) + 'vec' + '.npy'), pos_vec)


if __name__ == "__main__":

    args=parse_arguments()
    reformat_data(args)
