import sys
import os
import torch, torch.nn, torch.autograd
import numpy as np

from Atomic_HGNN import Atomic_HGNN
import joblib
import os.path as op
import pickle
import random

import time
import argparse
from torchvision import transforms
import glob
import cv2
from  Atomic_node_only_lstm import  Atomic_node_only_lstm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def collate_fn_atomic(batch):
    N = len(batch)
    max_node_num = 4
    sq_len=5

    head_batch = np.zeros((N, sq_len, 4, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))
    attmat_batch = np.zeros((N, sq_len, max_node_num, max_node_num))
    index_batch = np.zeros((N,  sq_len))

    for i, (head_patch_sq, pos_sq, attmat_sq, index_sq) in enumerate(batch):
            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq
            attmat_batch[i, ...] = attmat_sq
            index_batch[i,...]=index_sq

    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)
    index_batch = torch.FloatTensor(index_batch)

    return head_batch, pos_batch, attmat_batch, index_batch

def load_best_checkpoint(model,path):
    checkpoint_dir = path
    best_model_file = os.path.join(checkpoint_dir)
    if os.path.isfile(best_model_file):
        print("====> loading best model {}".format(best_model_file))
        checkpoint = torch.load(best_model_file, encoding='latin1')
        # print('===================================> checkpoint===================================> ')
        # print(checkpoint['state_dict'].keys())

        model_dict = model.state_dict()

        #
        # print('===================================> model ===================================> ')
        # print(model_dict.keys())

        pretrained_model = checkpoint['state_dict']
        pretrained_dict = {}
        for k, v in pretrained_model.items():
            if k[len('module.'):] in model_dict:
                pretrained_dict[k[len('module.'):]] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(model_dict)
        exit(0)
        #model.cuda()
        print("===> loaded best model {} (epoch {})".format(best_model_file, checkpoint['epoch']))
        return model

class mydataset_atomic_with_label_first_view(torch.utils.data.Dataset):
    def __init__(self, seq, args):
        self.seq = seq
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = args.img_path#'/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = args.tracker_bbox #'../../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = args.battery_bbox #'../../3d_pose2gaze/record_bbox/'
        self.obj_bbox = args.obj_bbox #'/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path =  args.obj_bbox #'/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = args.feature_single #'/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        rec = self.seq[index]
        head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224)) # [5,4,3,224,224]
        pos_sq = np.zeros((self.seq_size, self.node_num, 6)) #[5,4,6]
        hog_sq = np.zeros((self.seq_size, self.feature_dim*2))
        attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num)) #[5,4,4]
        clip, start_id, obj_list_seq, obj_list = rec
        img_names = sorted(glob.glob(os.path.join(self.img_path, clip.split('.')[0], 'kinect/*.jpg')))
        with open(self.person_tracker_bbox + clip, 'rb') as f:
            tracker_bbox = pickle.load(f, encoding='latin1')
        with open(self.person_battery_bbox + clip, 'rb') as f:
            battery_bbox = pickle.load(f, encoding='latin1')
        with open(self.feature_path + clip, 'rb') as f:
            features = pickle.load(f, encoding='latin1')

        for sq_id in range(len(obj_list_seq)):
            fid = start_id + sq_id
            img_name = img_names[fid]
            img = cv2.imread(img_name)
            for node_i in [0, 1]:
                if node_i == 0:
                    # if fid not in tracker_bbox:
                    #     continue
                    head_box = tracker_bbox[fid][0][0]
                    tracker_hog = features[1][fid][-self.feature_dim-10:-10]
                else:
                    # if fid not in battery_bbox:
                    #     continue
                    head_box = battery_bbox[fid][0][0]
                    battery_hog = features[2][fid][-self.feature_dim-10:-10]

                img_patch = img[int(head_box[1]):int(head_box[3]), int(head_box[0]): int(head_box[2])]
                img_patch = cv2.resize(img_patch, (224, 224))#.reshape((3, 224, 224))
                head = self.transforms(img_patch).numpy()
                for c in [0, 1, 2]:
                    head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
                pos_vec = np.array([head_box[0]/img.shape[1], head_box[1]/img.shape[0], head_box[2]/img.shape[1],
                            head_box[3]/img.shape[0], (head_box[0] + head_box[2])/2/img.shape[1], (head_box[1] + head_box[3])/2/img.shape[0]])
                # cv2.rectangle(img, (int(head_box[0]), int(head_box[1])),
                #                                         (int(head_box[2]), int(head_box[3])),
                #                                         (255, 0, 0), thickness=3)
                head_patch_sq[sq_id, node_i, ...] = head
                pos_sq[sq_id, node_i, :] = pos_vec
            hog_sq[sq_id, :] = np.hstack([tracker_hog, battery_hog])
            obj_id1, obj_id2, type1, type2, has_obj1, has_obj2 = obj_list_seq[sq_id]
            obj_name = obj_list[obj_id1]
            with open(self.obj_bbox + clip.split('.')[0] + '/' + str(obj_name) + '.p', 'rb') as f:
                obj_bbox = joblib.load(f)
            obj_curr = obj_bbox[fid]
            left = max(0, obj_curr[0])
            top = max(0, obj_curr[1])
            right = min(obj_curr[0] + obj_curr[2], img.shape[1])
            bottom = min(obj_curr[1] + obj_curr[3], img.shape[0])
            img_patch = img[int(top):int(bottom), int(left): int(right)]
            # print(img_patch.shape, obj_curr[0], obj_curr[0] + obj_curr[2], obj_curr[1], obj_curr[1] + obj_curr[3])
            img_patch = cv2.resize(img_patch, (224, 224))#.reshape((3, 224, 224))
            head = self.transforms(img_patch).numpy()
            for c in [0, 1, 2]:
                head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
            head_patch_sq[sq_id, 2, ...] = head
            pos_vec = np.array([obj_curr[0] / img.shape[1], obj_curr[1] / img.shape[0], (obj_curr[2] + obj_curr[0]) / img.shape[1],
                                (obj_curr[3] + obj_curr[1]) / img.shape[0], (obj_curr[0] + obj_curr[2] / 2) / img.shape[1],
                                (obj_curr[1] + obj_curr[3] / 2) / img.shape[0]])
            # cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
            #                                             (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
            #                                             (255, 0, 0), thickness=3)
            pos_sq[sq_id, 2, :] = pos_vec

            obj_name = obj_list[obj_id2]
            with open(self.obj_bbox + clip.split('.')[0] + '/' + str(obj_name) + '.p', 'rb') as f:
                obj_bbox = joblib.load(f)
            obj_curr = obj_bbox[fid]
            left = max(0, obj_curr[0])
            top = max(0, obj_curr[1])
            right = min(obj_curr[0] + obj_curr[2], img.shape[1])
            bottom = min(obj_curr[1] + obj_curr[3], img.shape[0])
            img_patch = img[int(top):int(bottom), int(left): int(right)]
            img_patch = cv2.resize(img_patch, (224, 224))#.reshape((3, 224, 224))
            head = self.transforms(img_patch).numpy()
            for c in [0, 1, 2]:
                head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
            head_patch_sq[sq_id, 3, ...] = head

            pos_vec = np.array(
                [obj_curr[0] / img.shape[1], obj_curr[1] / img.shape[0], (obj_curr[2] + obj_curr[0]) / img.shape[1],
                 (obj_curr[3] + obj_curr[1]) / img.shape[0], (obj_curr[0] + obj_curr[2] / 2) / img.shape[1],
                 (obj_curr[1] + obj_curr[3] / 2) / img.shape[0]])
            # cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
            #                                             (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
            #                                             (255, 0, 0), thickness=3)
            # cv2.imshow('img', img)
            # cv2.waitKey(20)
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

class mydataset_atomic_with_label_first_view_reformat(torch.utils.data.Dataset):

    def __init__(self, seq, args):
        self.seq = seq
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = args.img_path
        self.person_tracker_bbox = args.tracker_bbox
        self.person_battery_bbox = args.battery_bbox
        self.obj_bbox = args.obj_bbox
        self.cate_path = args.cate_path
        self.feature_path = args.feature_single

        self.tracker_head = args.data_path2 + 'tracker_head_patch/'
        self.battery_head = args.data_path2 + 'battery_head_patch/'
        self.obj_patch = args.data_path2 + 'obj_patch/'


    def __getitem__(self, index):
        rec = self.seq[index]
        head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224))  # [5,4,3,224,224]
        pos_sq = np.zeros((self.seq_size, self.node_num, 6))  # [5,4,6]
        hog_sq = np.zeros((self.seq_size, self.feature_dim * 2))
        attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num))  # [5,4,4]
        clip, start_id, obj_list_seq, obj_list = rec
        clip_pre=clip.split('.')[0]
        with open(self.feature_path + clip, 'rb') as f:
            features = pickle.load(f, encoding='latin1')

        for sq_id in range(len(obj_list_seq)):
            fid = start_id + sq_id
            for node_i in [0, 1]:
                if node_i == 0:
                    # if fid not in tracker_bbox:
                    #     continue
                    head_box = cv2.imread(os.path.join(self.tracker_head, clip_pre, str(fid) + 'head' + '.jpg'))
                    head_box = cv2.resize(head_box, (224, 224))  # .reshape((3, 224, 224))
                    head_box = self.transforms(head_box).numpy()
                    for c in [0, 1, 2]:
                        head_box[c, :, :] = (head_box[c, :, :] - 0.5) / 0.5
                    tracker_hog = features[1][fid][-self.feature_dim - 10:-10]
                    pos_vec = np.load(os.path.join(self.tracker_head, clip_pre, str(fid) + 'head_vec' + '.npy'))
                else:
                    # if fid not in battery_bbox:
                    #     continue
                    head_box = cv2.imread(os.path.join(self.battery_head, clip_pre, str(fid) + 'head' + '.jpg'))

                    head_box = cv2.resize(head_box, (224, 224))  # .reshape((3, 224, 224))
                    head_box = self.transforms(head_box).numpy()
                    for c in [0, 1, 2]:
                        head_box[c, :, :] = (head_box[c, :, :] - 0.5) / 0.5

                    battery_hog = features[2][fid][-self.feature_dim - 10:-10]
                    pos_vec = np.load(os.path.join(self.battery_head, clip_pre, str(fid) + 'head_vec' + '.npy'))

                head_patch_sq[sq_id, node_i, ...] = head_box
                pos_sq[sq_id, node_i, :] = pos_vec

            hog_sq[sq_id, :] = np.hstack([tracker_hog, battery_hog])
            obj_id1, obj_id2, type1, type2, has_obj1, has_obj2 = obj_list_seq[sq_id]

            obj_name = obj_list[obj_id1]
            obj = cv2.imread(os.path.join(self.obj_patch, clip_pre, str(obj_name) + '_' + str(fid) + '.jpg'))
            obj = cv2.resize(obj, (224, 224))  # .reshape((3, 224, 224))
            obj = self.transforms(obj).numpy()
            for c in [0, 1, 2]:
                obj[c, :, :] = (obj[c, :, :] - 0.5) / 0.5
            head_patch_sq[sq_id, 2, ...] = obj
            pos_vec = np.load(os.path.join(self.obj_patch, clip_pre, str(obj_name) + '_' + str(fid) + 'vec.npy'))
            pos_sq[sq_id, 2, :] = pos_vec

            obj_name = obj_list[obj_id2]
            obj = cv2.imread(os.path.join(self.obj_patch, clip_pre, str(obj_name) + '_' + str(fid) + '.jpg'))
            obj = cv2.resize(obj, (224, 224))  # .reshape((3, 224, 224))
            obj = self.transforms(obj).numpy()
            for c in [0, 1, 2]:
                obj[c, :, :] = (obj[c, :, :] - 0.5) / 0.5
            head_patch_sq[sq_id, 3, ...] = obj
            pos_vec = np.load(os.path.join(self.obj_patch, clip_pre, str(obj_name) + '_' + str(fid) + 'vec.npy'))
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

def collate_fn_atomic_first_view(batch):
    N = len(batch)
    max_node_num = 4
    sq_len=5

    head_batch = np.zeros((N, sq_len, 4, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))
    attmat_batch = np.zeros((N, sq_len, max_node_num, max_node_num))
    hog_batch = np.zeros((N, sq_len, 162*2))
    index_batch = np.zeros((N,  sq_len))
    label_batch = np.zeros(N)

    for i, (head_patch_sq, pos_sq, attmat_sq, index_sq, hog_sq) in enumerate(batch):
            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq
            attmat_batch[i, ...] = attmat_sq
            index_batch[i,...]=index_sq
            hog_batch[i, ...] = hog_sq

    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)
    index_batch = torch.FloatTensor(index_batch)
    hog_batch = torch.FloatTensor(hog_batch)

    return head_batch, pos_batch, attmat_batch, index_batch, hog_batch

class mydataset_atomic(torch.utils.data.Dataset):
    def __init__(self, seq, args):
        self.seq = seq
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.img_path = args.img_path
        self.person_tracker_bbox = args.tracker_bbox
        self.person_battery_bbox = args.battery_bbox
        self.obj_bbox =args.obj_bbox
        self.cate_path = args.cate_path

    def __getitem__(self, index):
        rec = self.seq[index]
        head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224)) # [5,4,3,224,224]
        pos_sq = np.zeros((self.seq_size, self.node_num, 6)) #[5,4,6]
        attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num)) #[5,4,4]
        clip, start_id, obj_list_seq, obj_list = rec
        img_names = sorted(glob.glob(os.path.join(self.img_path, clip.split('.')[0], 'kinect/*.jpg')))
        with open(self.person_tracker_bbox + clip, 'rb') as f:
            tracker_bbox = pickle.load(f,encoding='latin1')
        with open(self.person_battery_bbox + clip, 'rb') as f:
            battery_bbox = pickle.load(f, encoding='latin1')

        for sq_id in range(len(obj_list_seq)):
            fid = int(start_id + sq_id)
            img_name = img_names[fid]
            img = cv2.imread(img_name)
            for node_i in [0, 1]:
                if node_i == 0:
                    # if fid not in tracker_bbox:
                    #     continue
                    head_box = tracker_bbox[fid][0][0]
                else:
                    # if fid not in battery_bbox:
                    #     continue
                    head_box = battery_bbox[fid][0][0]
                img_patch = img[int(head_box[1]):int(head_box[3]), int(head_box[0]): int(head_box[2])]
                img_patch = cv2.resize(img_patch, (224, 224))#.reshape((3, 224, 224))
                head = self.transforms(img_patch).numpy()
                for c in [0, 1, 2]:
                    head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
                pos_vec = np.array([head_box[0]/img.shape[1], head_box[1]/img.shape[0], head_box[2]/img.shape[1],
                            head_box[3]/img.shape[0], (head_box[0] + head_box[2])/2/img.shape[1], (head_box[1] + head_box[3])/2/img.shape[0]])
                # cv2.rectangle(img, (int(head_box[0]), int(head_box[1])),
                #                                         (int(head_box[2]), int(head_box[3])),
                #                                         (255, 0, 0), thickness=3)
                head_patch_sq[sq_id, node_i, ...] = head
                pos_sq[sq_id, node_i, :] = pos_vec

            obj_id1, obj_id2, type1, type2, has_obj1, has_obj2 = obj_list_seq[sq_id]
            obj_name = obj_list[int(obj_id1)]
            with open(self.obj_bbox + clip.split('.')[0] + '/' + str(int(obj_name)) + '.p', 'rb') as f:
                obj_bbox = joblib.load(f)
            obj_curr = obj_bbox[fid]
            left = max(0, obj_curr[0])
            top = max(0, obj_curr[1])
            right = min(obj_curr[0] + obj_curr[2], img.shape[1])
            bottom = min(obj_curr[1] + obj_curr[3], img.shape[0])
            img_patch = img[int(top):int(bottom), int(left): int(right)]
            # print(img_patch.shape, obj_curr[0], obj_curr[0] + obj_curr[2], obj_curr[1], obj_curr[1] + obj_curr[3])
            img_patch = cv2.resize(img_patch, (224, 224))#.reshape((3, 224, 224))
            head = self.transforms(img_patch).numpy()
            for c in [0, 1, 2]:
                head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
            head_patch_sq[sq_id, 2, ...] = head
            pos_vec = np.array([obj_curr[0] / img.shape[1], obj_curr[1] / img.shape[0], (obj_curr[2] + obj_curr[0]) / img.shape[1],
                                (obj_curr[3] + obj_curr[1]) / img.shape[0], (obj_curr[0] + obj_curr[2] / 2) / img.shape[1],
                                (obj_curr[1] + obj_curr[3] / 2) / img.shape[0]])
            # cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
            #                                             (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
            #                                             (255, 0, 0), thickness=3)
            pos_sq[sq_id, 2, :] = pos_vec

            obj_name = obj_list[int(obj_id2)]
            with open(self.obj_bbox + clip.split('.')[0] + '/' + str(int(obj_name)) + '.p', 'rb') as f:
                obj_bbox = pickle.load(f,encoding='latin1')
            obj_curr = obj_bbox[fid]
            left = max(0, obj_curr[0])
            top = max(0, obj_curr[1])
            right = min(obj_curr[0] + obj_curr[2], img.shape[1])
            bottom = min(obj_curr[1] + obj_curr[3], img.shape[0])
            img_patch = img[int(top):int(bottom), int(left): int(right)]
            img_patch = cv2.resize(img_patch, (224, 224))#.reshape((3, 224, 224))
            head = self.transforms(img_patch).numpy()
            for c in [0, 1, 2]:
                head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
            head_patch_sq[sq_id, 3, ...] = head

            pos_vec = np.array(
                [obj_curr[0] / img.shape[1], obj_curr[1] / img.shape[0], (obj_curr[2] + obj_curr[0]) / img.shape[1],
                 (obj_curr[3] + obj_curr[1]) / img.shape[0], (obj_curr[0] + obj_curr[2] / 2) / img.shape[1],
                 (obj_curr[1] + obj_curr[3] / 2) / img.shape[0]])
            # cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
            #                                             (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
            #                                             (255, 0, 0), thickness=3)
            # cv2.imshow('img', img)
            # cv2.waitKey(20)
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

        return head_patch_sq, pos_sq, attmat_sq, index

    def __len__(self):
        return len(self.seq)


def test(test_loader, model, args):
    #atomic_label={0:'single', 1:'mutual', 2:'avert', 3:'refer', 4:'follow', 5:'share'}
    model.eval()
    test_results = []
    for i, (head_batch, pos_batch, attmat_batch, index_batch, hog_batch) in enumerate(test_loader):
        batch_size = head_batch.shape[0]
        test_results.append([])
        test_results[i].append(index_batch.numpy())
        test_results[i].append(attmat_batch.numpy())
        if args.cuda:
            heads = (torch.autograd.Variable(head_batch)).cuda()
            poses = (torch.autograd.Variable(pos_batch)).cuda()
            attmat_gt=(torch.autograd.Variable(attmat_batch)).cuda()
            hog = (torch.autograd.Variable(hog_batch)).cuda()
            index = (torch.autograd.Variable(index_batch)).cuda()
        else:
            heads = (torch.autograd.Variable(head_batch))
            poses = (torch.autograd.Variable(pos_batch))
            attmat_gt=(torch.autograd.Variable(attmat_batch))
            hog = (torch.autograd.Variable(hog_batch))
            index = (torch.autograd.Variable(index_batch))
        with torch.no_grad():
            pred_atomic = model(heads, poses, hog)
            for bid in range(batch_size):
                #print(pred_atomic.shape)
                pred = torch.argmax(pred_atomic[bid, :], dim=0)
                test_results[i].append(pred.data.cpu().numpy())
    return test_results

    #todo: save the test results

def check_type(curr_obj_1, obj_list, category, clip):
    obj_name = './post_box_reid/' + clip.split('.')[0] + '/' + str(obj_list[curr_obj_1]) + '.p'
    if category[obj_name][0] == 'person':
        return 'person'
    else:
        return 'object'

def check_atom(attmat, start_id, video_len, obj_list, category, clip, tracker_bbox, battery_bbox):
    obj_list_seq = []
    exist_obj_1 = dict()
    exist_obj_2 = dict()
    end = min(start_id + 5, video_len + 1)
    # print(start_id)
    for i in range(start_id, end):
        if not i in attmat:
            return False
        if not i in tracker_bbox:
            return False
        if not i in battery_bbox:
            return False
        # print(i, attmat[i][:2, :])
        curr_obj_1_arr = np.where(attmat[i][0, :] == 1)[0]
        curr_obj_2_arr = np.where(attmat[i][1, :] == 1)[0]
        if curr_obj_1_arr.shape[0] == 0:
            if len(exist_obj_1.keys()) < 1 and len(exist_obj_2.keys()) < 1:
                return False
            else:
                if len(exist_obj_1.keys()) > 0:
                    curr_obj_1 = exist_obj_1.keys()[0]
                else:
                    curr_obj_1 = exist_obj_2.keys()[0]
        else:
            curr_obj_1 = curr_obj_1_arr[0]

        if curr_obj_2_arr.shape[0] == 0:
            if len(exist_obj_1.keys()) < 1 and len(exist_obj_2.keys()) < 1:
                return False
            else:
                if len(exist_obj_2.keys()) > 0:
                    curr_obj_2 = exist_obj_2.keys()[0]
                else:
                    curr_obj_2 = exist_obj_1.keys()[0]
        else:
            curr_obj_2 = curr_obj_2_arr[0]

        if not curr_obj_1 in exist_obj_1:
            if len(exist_obj_1.keys()) < 1:
                obj_type = check_type(curr_obj_1, obj_list, category, clip)
                exist_obj_1[curr_obj_1] = obj_type
            else:
                obj_type = check_type(curr_obj_1, obj_list, category, clip)
                if obj_type == 'object':
                    return False
                else:
                    exist_obj_1[curr_obj_1] = obj_type

        if not curr_obj_2 in exist_obj_2:
            if len(exist_obj_2.keys()) < 1:
                obj_type = check_type(curr_obj_2, obj_list, category, clip)
                exist_obj_2[curr_obj_2] = obj_type
            else:
                obj_type = check_type(curr_obj_2, obj_list, category, clip)
                if obj_type == 'object':
                    return False
                else:
                    exist_obj_2[curr_obj_2] = obj_type

        obj_list_seq.append([curr_obj_1, curr_obj_2, exist_obj_1[curr_obj_1], exist_obj_2[curr_obj_2],
                             curr_obj_1_arr.shape[0], curr_obj_2_arr.shape[0]])
    return obj_list_seq

def find_test_seq(clip):
    attmat_path = '/home/shuwen/data/data_preprocessing2/record_attention_matrix/'
    cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
    person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
    person_battery_bbox = '../3d_pose2gaze/record_bbox/'
    slide_size = 1
    test_seq = []
    with open(attmat_path + clip, 'rb') as f:
        attmat_obj = pickle.load(f)
    with open(cate_path + clip.split('.')[0] + '/' + clip, 'rb') as f:
        category = joblib.load(f)
    with open(person_tracker_bbox + clip, 'rb') as f:
        tracker_bbox = joblib.load(f)
    with open(person_battery_bbox + clip, 'rb') as f:
        battery_bbox = joblib.load(f)
    attmat, obj_list = attmat_obj
    start_id = 0
    video_len = attmat.keys()[-1]
    while(start_id < video_len):
        obj_list_seq = check_atom(attmat, start_id, video_len, obj_list, category, clip, tracker_bbox, battery_bbox)
        if obj_list_seq:
            # print([clip, start_id, obj_list_seq, obj_list])
            test_seq.append([clip, start_id, obj_list_seq, obj_list])
            start_id = min(start_id + 5, video_len + 1)
        else:
            start_id += slide_size
    return test_seq

def visualize(clip):
    save_path = '/home/shuwen/data/data_preprocessing2/record_event_seq/'
    clip = clip.split('.')[0]
    img_path =  '/home/shuwen/data/data_preprocessing2/annotations/'
    person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
    person_battery_bbox = '../3d_pose2gaze/record_bbox/'
    obj_bbox_path = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
    atomic_label = {0: 'single', 1: 'mutual', 2: 'avert', 3: 'refer', 4: 'follow', 5: 'share'}
    with open(save_path + clip + '.p', 'rb') as f:
        test_seq, test_results = pickle.load(f)
    for i in range(len(test_results)):
        index_batch = test_results[i][0]
        attmat = test_results[i][1]
        for batch_id, batch in enumerate(index_batch):
            seq_id = batch[0]
            print(attmat[batch_id])
            raw_input('enter')
            seq = test_seq[int(seq_id)]
            clip, start_id, obj_list_seq, obj_list = seq
            img_names = sorted(glob.glob(os.path.join(img_path, clip.split('.')[0], 'kinect/*.jpg')))
            with open(person_tracker_bbox + clip, 'rb') as f:
                tracker_bbox = joblib.load(f)
            with open(person_battery_bbox + clip, 'rb') as f:
                battery_bbox = joblib.load(f)
            for sq_id in range(len(obj_list_seq)):
                fid = start_id + sq_id
                img_name = img_names[fid]
                img = cv2.imread(img_name)
                for node_i in [0, 1]:
                    if node_i == 0:
                        # if fid not in tracker_bbox:
                        #     continue
                        head_box = tracker_bbox[fid][0][0]
                    else:
                        # if fid not in battery_bbox:
                        #     continue
                        head_box = battery_bbox[fid][0][0]
                    cv2.rectangle(img, (int(head_box[0]), int(head_box[1])),
                                                            (int(head_box[2]), int(head_box[3])),
                                                            (255, 0, 0), thickness=3)
                obj_id1, obj_id2, type1, type2, has_obj1, has_obj2 = obj_list_seq[sq_id]
                obj_name = obj_list[obj_id1]
                with open(obj_bbox_path + clip.split('.')[0] + '/' + str(obj_name) + '.p', 'rb') as f:
                    obj_bbox = joblib.load(f)
                obj_curr = obj_bbox[fid]
                cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
                                                            (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
                                                            (255, 0, 0), thickness=3)
                obj_name = obj_list[obj_id2]
                with open(obj_bbox_path + clip.split('.')[0] + '/' + str(obj_name) + '.p', 'rb') as f:
                    obj_bbox = joblib.load(f)
                obj_curr = obj_bbox[fid]
                cv2.rectangle(img, (int(obj_curr[0]), int(obj_curr[1])),
                                                            (int(obj_curr[0] + obj_curr[2]), int(obj_curr[1] + obj_curr[3])),
                                                            (255, 0, 0), thickness=3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, 'seq_id:{}  type:{}'.format(seq_id, atomic_label[int(test_results[i][batch_id + 2])]), (org[0], org[1]), font,
                                  fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow('img', img)
                cv2.waitKey(20)


def main(args):

    # get test data
    attmat_path = args.attmat_path
    cate_path = args.cate_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    clips  = os.listdir(attmat_path)
    #clips = ['test_94342_14.p']
    # model
    model = Atomic_node_only_lstm(args)
    args.cuda = False
    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
    model = load_best_checkpoint(model, path='.')  # todo: path here
    for clip in clips[:30]:
        # if os.path.exists(save_path + clip):
        #     continue
        # print(clip)
        # test_seq = find_test_seq(clip)
        # test_set = mydataset_atomic(test_seq)
        # test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic, batch_size=16, shuffle=False)
        #
        # # ---------------------------------------------------------------------------------------------------------
        # # test
        # test_results = test(test_loader, model, args)
        # with open(save_path + clip, 'wb') as f:
        #     pickle.dump([test_seq, test_results], f)

        visualize(clip)


def parse_arguments():

    project_name = 'test atomic event'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--project-name', default=project_name, help='project name')
    # model settings
    parser.add_argument('--small-attr-class-num', type=int, default=7, help='small attribute class number')
    parser.add_argument('--big-attr-class-num', type=int, default=6, help='big attribute class number')
    parser.add_argument('--roi-feature-size',type=int, default=64*3, help='node and edge feature size')
    parser.add_argument('--message-size', type=int, default=12, help='message size of the message function')
    parser.add_argument('--lstm-seq-size', type=int, default=15, help='lstm sequence length')
    parser.add_argument('--lstm-hidden-size',type=int, default=500, help='hiddden state size of lstm')
    parser.add_argument('--link-hidden-size', type=int, default=1024, help='link hidden size of the link function')
    parser.add_argument('--link-hidden-layers', type=int, default=2, help='link hidden layers of the link function')
    parser.add_argument('--propagate-layers', type=int, default=2, help='propagate layers for message passing')


    home_path='/home/lfan/Dropbox/Projects/NIPS20/'
    home_path2='/media/lfan/HDD/NIPS20/'
    parser.add_argument('--project-path',default = home_path)
    parser.add_argument('--project-path2', default=home_path2)
    parser.add_argument('--data-path', default=home_path+'data/')
    parser.add_argument('--data-path2', default=home_path2 + 'data/')
    parser.add_argument('--img-path', default=home_path+'annotations/')
    parser.add_argument('--save-root', default='/media/lfan/HDD/NIPS20/Result/')
    parser.add_argument('--save-path', default='/media/lfan/HDD/NIPS20/Result/')
    parser.add_argument('--init-cps', default=home_path+'data/cps_comb1.p')
    parser.add_argument('--stat-path', default=home_path+'data/stat/')
    parser.add_argument('--attmat-path', default=home_path+'data/record_attention_matrix/')
    parser.add_argument('--cate-path', default=home_path2+'data/track_cate/')
    parser.add_argument('--tracker-bbox', default=home_path2+'data/tracker_record_bbox/')
    parser.add_argument('--battery-bbox', default=home_path2+'data/record_bbox/')
    parser.add_argument('--obj-bbox', default=home_path2+'data/post_neighbor_smooth_newseq/')
    parser.add_argument('--seg-label', default=home_path + 'data/segment_labels/')
    parser.add_argument('--event-model-path', default=home_path+'code/model_event.pth')
    parser.add_argument('--atomic-event-path', default=home_path+'code/model_best_atomic.pth')
    parser.add_argument('--feature-single', default=home_path2 + 'data/feature_single/')


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
