import numpy as np
import torch
from torch.autograd import Variable
import pickle

from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import os
from sklearn import metrics
import glob
from annotation_clean import *
from sklearn import svm
import matplotlib.pyplot as plt
import joblib
# from mind_model import *
from mind_model_att import *
import seaborn as sn
import random
from torchvision import transforms
import cv2
import sys
sys.path.append('./data_processing_scripts/')
from metadata import *
sys.path.append('./attention_classifier/')

def print_result():
    with open('./cptk_combined_att_add_1024/seq_init_seg_0.p', 'rb') as f:
        results = pickle.load(f)

    print(results[0][0])
    print(results[0][1])
    print(results[0][2])
    print(results[0][3])
    print(results[0][4])

def plot_confusion_matrix(cmc):
    df_cm = pd.DataFrame(cmc, range(cmc.shape[0]), range(cmc.shape[1]))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

def calculate_dis(data_path, model_type):
    clips = os.listdir(data_path)
    train_x = {'mc':[0, 0, 0, 0], 'm1':[0, 0, 0, 0], 'm2':[0, 0, 0, 0], 'm12':[0, 0, 0, 0], 'm21':[0, 0, 0, 0]}
    test_x = {'mc': [0, 0, 0, 0], 'm1': [0, 0, 0, 0], 'm2': [0, 0, 0, 0], 'm12': [0, 0, 0, 0], 'm21': [0, 0, 0, 0]}
    for clip in clips:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            if model_type == 'single' or model_type == 'single_one_hot':
                vec_input, label_, _ = pickle.load(f)
            else:
                vec_input, label_ = pickle.load(f)

            for lid, label in enumerate(label_):
                label = label.reshape(-1)
                labels_mc = np.argmax(label[:4])
                labels_m21 = np.argmax(label[4:8])
                labels_m12 = np.argmax(label[8:12])
                labels_m1 = np.argmax(label[12:16])
                labels_m2 = np.argmax(label[16:20])

                if clip in mind_test_clips:
                    test_x['mc'][labels_mc] += 1
                    test_x['m1'][labels_m1] += 1
                    test_x['m2'][labels_m2] += 1
                    test_x['m12'][labels_m12] += 1
                    test_x['m21'][labels_m21] += 1
                else:
                    train_x['mc'][labels_mc] += 1
                    train_x['m1'][labels_m1] += 1
                    train_x['m2'][labels_m2] += 1
                    train_x['m12'][labels_m12] += 1
                    train_x['m21'][labels_m21] += 1
    print(train_x)
    print(test_x)

def get_data(data_path, model_type):

    clips = os.listdir(data_path)

    data_t, data_v, data_test = [], [], []
    labels_t, labels_v, labels_test = [], [], []
    random.seed(1234)
    random.shuffle(clips)
    clip_ratio1 = int(len(clips)*0.6)
    clip_ratio2 = int(len(clips) * 0.2)
    for clip in clips[:clip_ratio1]:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            vec_input, label_, _, _ = pickle.load(f)
            data_t += vec_input
            labels_t += label_

    for clip in clips[clip_ratio1:clip_ratio1 + clip_ratio2]:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            vec_input, label_, _, _ = pickle.load(f)
            data_v += vec_input
            labels_v += label_

    for clip in clips[clip_ratio1 + clip_ratio2:]:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            vec_input, label_, _, _ = pickle.load(f)
            data_test += vec_input
            labels_test += label_

    train_x, train_y = data_t, labels_t
    validate_x, validate_y = data_v, labels_v
    test_x, test_y = data_test, labels_test
    # new_data = []
    # new_label = []
    # for lid, label in enumerate(labels):
    #     # if label == 1023:
    #     #     continue
    #     new_data.append(data[lid])
    #     new_label.append(labels[lid])
    #
    # c = list(zip(new_data, new_label))
    #
    # random.shuffle(c)
    # train_ratio = int(len(c) * 0.6)
    # validate_ratio = int(len(c) * 0.2)
    # data, label = zip(*c)
    #
    # train_x, train_y = data[:train_ratio], label[:train_ratio]
    # validate_x, validate_y = data[train_ratio:train_ratio + validate_ratio], label[train_ratio:train_ratio + validate_ratio]
    # test_x, test_y = data[train_ratio + validate_ratio:], label[train_ratio + validate_ratio:]
    print(len(train_y), len(validate_y), len(test_y))
    mind_count = np.zeros(1024)
    no_event = 0
    event = 0
    for data_id in range(len(train_y)):
        p1_event, p2_event, _, _, _ = train_x[data_id]
        if np.all(p1_event == 0) and np.all(p2_event) == 0:
            assert train_y[data_id] == 1023
            no_event += 1
        else:
            event += 1
        mind_count[train_y[data_id]] += 1
    print(event, no_event)
    return train_x, train_y, validate_x, validate_y, test_x, test_y, mind_count

def get_data_balance(data_path, model_type):

    clips = os.listdir(data_path)

    data_t, data_v, data_test = [], [], []
    labels_t, labels_v, labels_test = [], [], []
    random.seed(1234)
    random.shuffle(clips)
    clip_ratio1 = int(len(clips) * 0.6)
    clip_ratio2 = int(len(clips) * 0.2)
    for clip in clips[:clip_ratio1]:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            vec_input, label_, _, _ = pickle.load(f)
            data_t += vec_input
            labels_t += label_

    for clip in clips[clip_ratio1:clip_ratio1 + clip_ratio2]:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            vec_input, label_, _, _ = pickle.load(f)
            data_v += vec_input
            labels_v += label_

    for clip in clips[clip_ratio1 + clip_ratio2:]:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            vec_input, label_, _, _ = pickle.load(f)
            data_test += vec_input
            labels_test += label_

    mind_count = np.zeros(1024)
    data_dict = {}
    for data_id in range(len(data_t)):
        p1_event, p2_event, _, _, _ = data_t[data_id]
        mind_count[labels_t[data_id]] += 1
        if labels_t[data_id] not in data_dict:
            data_dict[labels_t[data_id]] = []
        data_dict[labels_t[data_id]].append(data_t[data_id])

    count_max = np.max(mind_count)
    new_train_x = []
    new_train_y = []
    for key in data_dict.keys():
        length = len(data_dict[key])
        if length > 0:
            for temp in range(int(count_max)/length):
                new_train_x.extend(data_dict[key])
                temp_y = np.ones(length)*key
                new_train_y.extend(list(temp_y))

    c = list(zip(new_train_x, new_train_y))
    random.shuffle(c)
    data_t, labels_t = zip(*c)
    train_x, train_y = data_t, labels_t
    validate_x, validate_y = data_v, labels_v
    test_x, test_y = data_test, labels_test

    print(len(train_y), len(validate_y), len(test_y))

    mind_count = np.zeros(1024)
    for data_id in range(len(train_x)):
        mind_count[int(train_y[data_id])] += 1
    plt.plot(mind_count)
    plt.show()
    return train_x, train_y, validate_x, validate_y, test_x, test_y, mind_count

class mydataset_combined_att(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.transforms = transforms.Compose(
                            [transforms.Resize([224, 224]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        p1_event, p2_event, memory, indicator, att = self.train_x[index]
        # p1_event = np.exp(p1_event)/np.exp(p1_event).sum()
        # p2_event = np.exp(p2_event)/np.exp(p2_event).sum()
        # print(p1_event.shape, p2_event.shape, memory.shape, indicator.shape, att.shape)
        event_input = np.hstack([p1_event, p2_event, memory, indicator, att])

        actual_val = self.train_y[index]

        return event_input, actual_val

    def __len__(self):
        return len(self.train_x)

def collate_fn_combined_att(batch):
    N = len(batch)

    event_batch = np.zeros((N, 19))
    label_batch = np.zeros(N)

    for i, (event, label) in enumerate(batch):
        event_batch[i, ...] = event
        label_batch[i,...] = label

    event_batch = torch.FloatTensor(event_batch)
    label_batch = torch.LongTensor(label_batch)


    return event_batch, label_batch

class mydataset_combined(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.transforms = transforms.Compose(
                            [transforms.Resize([224, 224]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        p1_event, p2_event, memory, indicator = self.train_x[index]
        # p1_event = np.exp(p1_event)/np.exp(p1_event).sum()
        # p2_event = np.exp(p2_event)/np.exp(p2_event).sum()
        event_input = np.hstack([p1_event, p2_event, memory, indicator])

        actual_val = self.train_y[index]

        return event_input, actual_val

    def __len__(self):
        return len(self.train_x)

def collate_fn_combined(batch):
    N = len(batch)

    event_batch = np.zeros((N, 16))
    label_batch = np.zeros(N)

    for i, (event, label) in enumerate(batch):
        event_batch[i, ...] = event
        label_batch[i,...] = label

    event_batch = torch.FloatTensor(event_batch)
    label_batch = torch.LongTensor(label_batch)


    return event_batch, label_batch

class mydataset(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.transforms = transforms.Compose(
                            [transforms.Resize([224, 224]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        p1_event, p2_event, memory, indicator, hog, obj_patch = self.train_x[index]
        p1_event = np.exp(p1_event)/np.exp(p1_event).sum()
        p2_event = np.exp(p2_event)/np.exp(p2_event).sum()
        event_input = np.hstack([p1_event, p2_event, memory[1:], indicator[1:]])
        obj_patch = transforms.ToPILImage()(obj_patch)
        obj_patch = self.transforms(obj_patch)

        actual_val = self.train_y[index]
        labels_mc = np.argmax(actual_val[:4])
        labels_m21 = np.argmax(actual_val[4:8])
        labels_m12 = np.argmax(actual_val[8:12])
        labels_m1 = np.argmax(actual_val[12:16])
        labels_m2 = np.argmax(actual_val[16:20])

        return event_input, obj_patch, hog, labels_mc, labels_m1, labels_m2, labels_m12, labels_m21

    def __len__(self):
        return len(self.train_x)

def collate_fn(batch):
    N = len(batch)

    event_batch = np.zeros((N, 16))
    obj_batch = np.zeros((N, 3, 224, 224))
    hog_batch = np.zeros((N, 162*2))
    mc_label_batch = np.zeros(N)
    m1_label_batch = np.zeros(N)
    m2_label_batch = np.zeros(N)
    m12_label_batch = np.zeros(N)
    m21_label_batch = np.zeros(N)

    for i, (event, obj, hog, mc, m1, m2, m12, m21) in enumerate(batch):
        event_batch[i, ...] = event
        obj_batch[i, ...] = obj
        hog_batch[i, ...] = hog
        mc_label_batch[i,...] = mc
        m1_label_batch[i, ...] = m1
        m2_label_batch[i, ...] = m2
        m12_label_batch[i, ...] = m12
        m21_label_batch[i, ...] = m21

    event_batch = torch.FloatTensor(event_batch)
    obj_batch = torch.FloatTensor(obj_batch)
    hog_batch = torch.FloatTensor(hog_batch)
    mc_label_batch = torch.LongTensor(mc_label_batch)
    m1_label_batch = torch.LongTensor(m1_label_batch)
    m2_label_batch = torch.LongTensor(m2_label_batch)
    m12_label_batch = torch.LongTensor(m12_label_batch)
    m21_label_batch = torch.LongTensor(m21_label_batch)

    return event_batch, obj_batch, hog_batch, mc_label_batch, m1_label_batch, m2_label_batch, m12_label_batch, m21_label_batch

class mydataset_lstm(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.transforms = transforms.Compose(
                            [transforms.Resize([224, 224]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        event_input, hog_input, img_input, box_input = self.train_x[index]
        obj_patch_input = np.zeros((self.seq_size, 3, 224, 224))


        for i in range(len(img_input)):

            # obj_patch
            img = cv2.imread(img_input[i])
            x_min, y_min, x_max, y_max = box_input[i]

            obj_patch = img[y_min:y_max, x_min:x_max]
            obj_patch = transforms.ToPILImage()(obj_patch)
            obj_patch = self.transforms(obj_patch).numpy()
            obj_patch_input[i, ...] = obj_patch

        actual_val = self.train_y[index]
        labels_mc = np.argmax(actual_val[:4])
        labels_m21 = np.argmax(actual_val[4:8])
        labels_m12 = np.argmax(actual_val[8:12])
        labels_m1 = np.argmax(actual_val[12:16])
        labels_m2 = np.argmax(actual_val[16:20])

        return event_input, obj_patch_input, hog_input, labels_mc, labels_m1, labels_m2, labels_m12, labels_m21

    def __len__(self):
        return len(self.train_x)

def collate_fn_lstm(batch):
    N = len(batch)
    seq_len = 5

    event_batch = np.zeros((N, seq_len, 16))
    obj_batch = np.zeros((N, seq_len, 3, 224, 224))
    hog_batch = np.zeros((N, seq_len, 162*2))
    mc_label_batch = np.zeros(N)
    m1_label_batch = np.zeros(N)
    m2_label_batch = np.zeros(N)
    m12_label_batch = np.zeros(N)
    m21_label_batch = np.zeros(N)

    for i, (event, obj, hog, mc, m1, m2, m12, m21) in enumerate(batch):
        event_batch[i, ...] = event
        obj_batch[i, ...] = obj
        hog_batch[i, ...] = hog
        mc_label_batch[i,...] = mc
        m1_label_batch[i, ...] = m1
        m2_label_batch[i, ...] = m2
        m12_label_batch[i, ...] = m12
        m21_label_batch[i, ...] = m21

    event_batch = torch.FloatTensor(event_batch)
    obj_batch = torch.FloatTensor(obj_batch)
    hog_batch = torch.FloatTensor(hog_batch)
    mc_label_batch = torch.LongTensor(mc_label_batch)
    m1_label_batch = torch.LongTensor(m1_label_batch)
    m2_label_batch = torch.LongTensor(m2_label_batch)
    m12_label_batch = torch.LongTensor(m12_label_batch)
    m21_label_batch = torch.LongTensor(m21_label_batch)
    return event_batch, obj_batch, hog_batch, mc_label_batch, m1_label_batch, m2_label_batch, m12_label_batch, m21_label_batch

class mydataset_lstm_cnn(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.transforms = transforms.Compose(
                            [transforms.Resize([224, 224]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        event_input, hog_input, img_input, box_input = self.train_x[index]
        obj_patch_input = np.zeros((self.seq_size, 3, 224, 224))


        for i in range(len(img_input)):

            # obj_patch
            img = cv2.imread(img_input[i])
            obj_patch = transforms.ToPILImage()(img)
            obj_patch = self.transforms(obj_patch).numpy()
            obj_patch_input[i, ...] = obj_patch

        actual_val = self.train_y[index]
        labels_mc = np.argmax(actual_val[:4])
        labels_m21 = np.argmax(actual_val[4:8])
        labels_m12 = np.argmax(actual_val[8:12])
        labels_m1 = np.argmax(actual_val[12:16])
        labels_m2 = np.argmax(actual_val[16:20])

        return obj_patch_input, hog_input, labels_mc, labels_m1, labels_m2, labels_m12, labels_m21

    def __len__(self):
        return len(self.train_x)

def collate_fn_lstm_cnn(batch):
    N = len(batch)
    seq_len = 5

    obj_batch = np.zeros((N, seq_len, 3, 224, 224))
    hog_batch = np.zeros((N, seq_len, 162*2))
    mc_label_batch = np.zeros(N)
    m1_label_batch = np.zeros(N)
    m2_label_batch = np.zeros(N)
    m12_label_batch = np.zeros(N)
    m21_label_batch = np.zeros(N)

    for i, (obj, hog, mc, m1, m2, m12, m21) in enumerate(batch):
        obj_batch[i, ...] = obj
        hog_batch[i, ...] = hog
        mc_label_batch[i,...] = mc
        m1_label_batch[i, ...] = m1
        m2_label_batch[i, ...] = m2
        m12_label_batch[i, ...] = m12
        m21_label_batch[i, ...] = m21

    obj_batch = torch.FloatTensor(obj_batch)
    hog_batch = torch.FloatTensor(hog_batch)
    mc_label_batch = torch.LongTensor(mc_label_batch)
    m1_label_batch = torch.LongTensor(m1_label_batch)
    m2_label_batch = torch.LongTensor(m2_label_batch)
    m12_label_batch = torch.LongTensor(m12_label_batch)
    m21_label_batch = torch.LongTensor(m21_label_batch)
    return obj_batch, hog_batch, mc_label_batch, m1_label_batch, m2_label_batch, m12_label_batch, m21_label_batch

class mydataset_cnn(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.transforms = transforms.Compose(
                            [transforms.Resize([224, 224]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        img_input, hog_input = self.train_x[index]
        obj_patch_input = np.zeros((3, 224, 224))

        img = cv2.imread(img_input)

        obj_patch = transforms.ToPILImage()(img)
        obj_patch = self.transforms(obj_patch).numpy()
        obj_patch_input = obj_patch

        actual_val = self.train_y[index]
        labels_mc = np.argmax(actual_val[:4])
        labels_m21 = np.argmax(actual_val[4:8])
        labels_m12 = np.argmax(actual_val[8:12])
        labels_m1 = np.argmax(actual_val[12:16])
        labels_m2 = np.argmax(actual_val[16:20])

        return obj_patch_input, hog_input, labels_mc, labels_m1, labels_m2, labels_m12, labels_m21

    def __len__(self):
        return len(self.train_x)

def collate_fn_cnn(batch):
    N = len(batch)

    obj_batch = np.zeros((N, 3, 224, 224))
    hog_batch = np.zeros((N, 162*2))
    mc_label_batch = np.zeros(N)
    m1_label_batch = np.zeros(N)
    m2_label_batch = np.zeros(N)
    m12_label_batch = np.zeros(N)
    m21_label_batch = np.zeros(N)

    for i, (obj, hog, mc, m1, m2, m12, m21) in enumerate(batch):
        obj_batch[i, ...] = obj
        hog_batch[i, ...] = hog
        mc_label_batch[i,...] = mc
        m1_label_batch[i, ...] = m1
        m2_label_batch[i, ...] = m2
        m12_label_batch[i, ...] = m12
        m21_label_batch[i, ...] = m21

    obj_batch = torch.FloatTensor(obj_batch)
    hog_batch = torch.FloatTensor(hog_batch)
    mc_label_batch = torch.LongTensor(mc_label_batch)
    m1_label_batch = torch.LongTensor(m1_label_batch)
    m2_label_batch = torch.LongTensor(m2_label_batch)
    m12_label_batch = torch.LongTensor(m12_label_batch)
    m21_label_batch = torch.LongTensor(m21_label_batch)
    return obj_batch, hog_batch, mc_label_batch, m1_label_batch, m2_label_batch, m12_label_batch, m21_label_batch

class mydataset_raw_feature(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.transforms = transforms.Compose(
                            [transforms.Resize([224, 224]),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        input = self.train_x[index]

        actual_val = self.train_y[index]
        labels_mc = np.argmax(actual_val[:4])
        labels_m21 = np.argmax(actual_val[4:8])
        labels_m12 = np.argmax(actual_val[8:12])
        labels_m1 = np.argmax(actual_val[12:16])
        labels_m2 = np.argmax(actual_val[16:20])

        return input, labels_mc, labels_m1, labels_m2, labels_m12, labels_m21

    def __len__(self):
        return len(self.train_x)

def collate_fn_raw_feature(batch):
    N = len(batch)

    event_batch = np.zeros((N, 16))
    obj_batch = np.zeros((N, 3, 224, 224))
    hog_batch = np.zeros((N, 162*2))
    mc_label_batch = np.zeros(N)
    m1_label_batch = np.zeros(N)
    m2_label_batch = np.zeros(N)
    m12_label_batch = np.zeros(N)
    m21_label_batch = np.zeros(N)

    for i, (input, mc, m1, m2, m12, m21) in enumerate(batch):
        event_batch[i, ...] = input
        mc_label_batch[i,...] = mc
        m1_label_batch[i, ...] = m1
        m2_label_batch[i, ...] = m2
        m12_label_batch[i, ...] = m12
        m21_label_batch[i, ...] = m21

    event_batch = torch.FloatTensor(event_batch)
    mc_label_batch = torch.LongTensor(mc_label_batch)
    m1_label_batch = torch.LongTensor(m1_label_batch)
    m2_label_batch = torch.LongTensor(m2_label_batch)
    m12_label_batch = torch.LongTensor(m12_label_batch)
    m21_label_batch = torch.LongTensor(m21_label_batch)

    return event_batch, mc_label_batch, m1_label_batch, m2_label_batch, m12_label_batch, m21_label_batch

def main():

    # data_path = '/home/shuwen/data/data_preprocessing2/mind_training_add_hog/'
    data_path = './mind_training_full_0915/'
    model_type = 'new_att'

    # calculate_dis(data_path, model_type)
    train_x, train_y, validate_x, validate_y, test_x, test_y, mind_count = get_data(data_path, model_type)
    print(len(train_x), len(validate_x), len(test_x))

    learningRate = 0.01
    epochs = 500
    batch_size = 256
    test_flag = 0
    if test_flag == 0:
        train_combined(model_type, learningRate, epochs, batch_size, train_x, train_y, validate_x, validate_y, mind_count) #, checkpoint='./cptk_single/model_best.pth')
    else:
        net = MLP_Combined_Att()
        net.load_state_dict(torch.load('./cptk_new_att/model_best.pth'))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        test_score(net, test_x, test_y, batch_size, model_type, 'test')

    # test_data_seq_tree_search_per_clip('tree_search')
    # test_data_seq_tree_search('tree_search_frame')
    # test_data_seq_tree_search('tree_search_init')
    # test_data_seq_tree_search('tree_search_unif_event')
    # test_data_seq_tree_search('tree_search_event_likelihood')
    # test_data_seq_per_clip('combined_att')
    # test_data_seq_ablation('combined_att_add_1024')

def test_score(net, data, label, batch_size, proj_name = None, dataset = None):

    net.eval()
    total_mc = np.empty(0)
    total_m21 = np.empty(0)
    total_m12 = np.empty(0)
    total_m2 = np.empty(0)
    total_m1 = np.empty(0)

    total_act_mc = np.empty(0)
    total_act_m21 = np.empty(0)
    total_act_m12 = np.empty(0)
    total_act_m2 = np.empty(0)
    total_act_m1 = np.empty(0)

    if proj_name == 'single' or proj_name == 'single_one_hot' or proj_name == 'tree_search':
        train_set = mydataset(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size,
                                                   shuffle=False)
    elif proj_name == 'cnn' or proj_name == 'cnn_no_hog':
        train_set = mydataset_cnn(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_cnn, batch_size=batch_size,
                                                   shuffle=False)
    elif proj_name == 'event_memory' or proj_name == 'combined' or proj_name == 'combined_event':
        train_set = mydataset_combined(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_combined, batch_size=batch_size,
                                                   shuffle=False)
    elif proj_name == 'combined_att' or proj_name == 'combined_att_add_1024' or proj_name == 'new_att' or proj_name == 'new_att_balance':
        train_set = mydataset_combined_att(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_combined_att, batch_size=batch_size,
                                                   shuffle=False)
    else:
        train_set = mydataset_lstm(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_lstm, batch_size=batch_size,
                                                   shuffle=False)


    net.eval()
    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    pbar = tqdm(train_loader)
    for batch in pbar:
        event_batch, label_batch = batch
        event_batch = event_batch.cuda()
        label_batch = label_batch.numpy()



        m = net(event_batch)
        max_score, idx_m = torch.max(m, 1)
        idx_m = idx_m.cpu().numpy()
        for data_id, data in enumerate(idx_m):
            pred_combination = mind_combination[data]

            total_m1 = np.append(total_m1, pred_combination[0])
            total_m2 = np.append(total_m2, pred_combination[1])
            total_m12 = np.append(total_m12, pred_combination[2])
            total_m21 = np.append(total_m21, pred_combination[3])
            total_mc = np.append(total_mc, pred_combination[4])

            true_combination = mind_combination[label_batch[data_id]]
            total_act_mc = np.append(total_act_mc, true_combination[4])
            total_act_m21 = np.append(total_act_m21, true_combination[3])
            total_act_m12 = np.append(total_act_m12, true_combination[2])
            total_act_m1 = np.append(total_act_m1, true_combination[0])
            total_act_m2 = np.append(total_act_m2, true_combination[1])

    if dataset:
        results_mc = metrics.classification_report(total_act_mc, total_mc, digits=3)
        results_m1 = metrics.classification_report(total_act_m1, total_m1, digits=3)
        results_m2 = metrics.classification_report(total_act_m2, total_m2, digits=3)
        results_m12 = metrics.classification_report(total_act_m12, total_m12, digits=3)
        results_m21 = metrics.classification_report(total_act_m21, total_m21, digits=3)

        print(results_mc)
        print(results_m1)
        print(results_m2)
        print(results_m12)
        print(results_m21)

        cmc = metrics.confusion_matrix(total_act_mc, total_mc)
        cm1 = metrics.confusion_matrix(total_act_m1, total_m1)
        cm2 = metrics.confusion_matrix(total_act_m2, total_m2)
        cm12 = metrics.confusion_matrix(total_act_m12, total_m12)
        cm21 = metrics.confusion_matrix(total_act_m21, total_m21)

        plot_confusion_matrix(cmc)
        plot_confusion_matrix(cm1)
        plot_confusion_matrix(cm2)
        plot_confusion_matrix(cm12)
        plot_confusion_matrix(cm21)

        score1 = metrics.f1_score(total_act_mc, total_mc, average='macro')
        score2 = metrics.f1_score(total_act_m1, total_m1, average='macro')
        score3 = metrics.f1_score(total_act_m2, total_m2, average='macro')
        score4 = metrics.f1_score(total_act_m12, total_m12, average='macro')
        score5 = metrics.f1_score(total_act_m21, total_m21, average='macro')
        print([score1, score2, score3, score4, score5])
        with open('./cptk_' + proj_name + '/' + dataset + '.p', 'wb') as f:
            pickle.dump([results_m1, results_m2, results_m12, results_m21, results_mc], f)

    # score1 = metrics.accuracy_score(total_act_mc, total_mc)
    # score2 = metrics.accuracy_score(total_act_m1, total_m1)
    # score3 = metrics.accuracy_score(total_act_m2, total_m2)
    # score4 = metrics.accuracy_score(total_act_m12, total_m12)
    # score5 = metrics.accuracy_score(total_act_m21, total_m21)
    score1 = metrics.f1_score(total_act_mc, total_mc, average='macro')
    score2 = metrics.f1_score(total_act_m1, total_m1, average='macro')
    score3 = metrics.f1_score(total_act_m2, total_m2, average='macro')
    score4 = metrics.f1_score(total_act_m12, total_m12, average='macro')
    score5 = metrics.f1_score(total_act_m21, total_m21, average='macro')
    return [score1, score2, score3, score4, score5]

def train_combined(save_prefix, learningRate, epochs, batch_size, train_x, train_y, validate_x, validate_y, mind_count, checkpoint = None, startepoch = None):
    if save_prefix == 'single' or save_prefix == 'tree_search':
        model = MindHog()
    elif save_prefix == 'single_one_hot':
        model = MindHog()
    elif save_prefix == 'cnn':
        model = MLP_Feature_Mem()
    elif save_prefix == 'cnn_no_hog':
        model = MindCNNNoHog()
    elif save_prefix == 'event_memory':
        model = MLP_Event_Memory()
    elif save_prefix == 'combined' or save_prefix == 'combined_event':
        model = MLP_Combined()
    elif save_prefix == 'combined_att' or save_prefix == 'combined_att_add_1024' or save_prefix == 'new_att' or save_prefix == 'new_att_balance':
        model = MLP_Combined_Att()
    else:
        model = MindLSTMHog()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    if startepoch is not None:
        startepoch = startepoch
    else:
        startepoch = 0
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    # mind_count[mind_count > 0] = 1./mind_count[mind_count > 0]
    # weights = torch.tensor(mind_count).float().cuda()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    losses = []
    best_score = 0

    if save_prefix == 'single' or save_prefix == 'single_one_hot' or save_prefix == 'tree_search':
        train_set = mydataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size,
                                                   shuffle=False)
    elif save_prefix == 'cnn' or save_prefix == 'cnn_no_hog':
        train_set = mydataset_cnn(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_cnn, batch_size=batch_size,
                                                   shuffle=False)
    elif save_prefix == 'event_memory' or save_prefix == 'combined' or save_prefix == 'combined_event':
        train_set = mydataset_combined(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_combined, batch_size=batch_size,
                                                   shuffle=False)
    elif save_prefix == 'combined_att' or save_prefix == 'combined_att_add_1024' or save_prefix == 'new_att' or save_prefix == 'new_att_balance':
        train_set = mydataset_combined_att(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_combined_att, batch_size=batch_size,
                                                   shuffle=False)
    else:
        train_set = mydataset_lstm(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_lstm, batch_size=batch_size,
                                                   shuffle=False)

    for epoch in range(startepoch, epochs):
        model.train()
        # training set -- perform model training
        epoch_training_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            event_batch, label_batch = batch
            event_batch = event_batch.cuda()
            label_batch = label_batch.cuda()

            # Make gradients zero for parameters 'W', 'b'
            optimizer.zero_grad()

            # m1, m2, m12, m21, mc = model(obj_batch)
            label = model(event_batch)
            loss = criterion(label, label_batch)
            loss.backward()

            optimizer.step()

            # calculating loss
            epoch_training_loss += loss.data.item()
            num_batches += 1

        # scheduler.step()
        print("epoch:{}/loss:{}".format(epoch, epoch_training_loss/num_batches))
        losses.append(epoch_training_loss/num_batches)

        score = test_score(model, validate_x, validate_y, batch_size, save_prefix)
        if sum(score) > best_score:
            best_score = sum(score)
            print('best_score: mc:{}, m1:{}, m2:{}, m12:{}, m21:{}'.format(score[0], score[1], score[2], score[3], score[4]))
            save_path = './cptk_' + save_prefix + '/model_best.pth'
            torch.save(model.state_dict(), save_path)
        if epoch%100 == 0:
            for param in optimizer.param_groups:
                if param['lr'] > 1e-5:
                    param['lr'] = param['lr']*0.5

        if epoch%50 == 0:
            save_path = './cptk_' + save_prefix + '/model_' +str(epoch)+'.pth'
            torch.save(model.state_dict(), save_path)
            # plt.plot(losses)
            # plt.show()


def check_overlap_return_area(head_box, obj_curr):
    max_left = max(head_box[0], obj_curr[0])
    max_top = max(head_box[1], obj_curr[1])
    min_right = min(head_box[2], obj_curr[2])
    min_bottom = min(head_box[3], obj_curr[3])
    if (min_right - max_left) > 0 and (min_bottom - max_top) > 0:
        return (min_right - max_left)*(min_bottom - max_top)
    return -100

def get_obj_name(obj_bbox, annt, frame_id):
    obj_candidates = annt.loc[annt.frame == frame_id]
    max_overlap = 0
    max_name = None
    max_bbox = None
    obj_bbox = [obj_bbox[0], obj_bbox[1], obj_bbox[0] + obj_bbox[2], obj_bbox[1] + obj_bbox[3]]
    obj_area = (obj_bbox[2] - obj_bbox[0])*(obj_bbox[3] - obj_bbox[1])
    for index, obj_candidate in obj_candidates.iterrows():
        if obj_candidate['name'].startswith('P'):
            continue
        candidate_bbox = [obj_candidate['x_min'], obj_candidate['y_min'], obj_candidate['x_max'], obj_candidate['y_max']]
        overlap = check_overlap_return_area(obj_bbox, candidate_bbox)
        if overlap > max_overlap and overlap/obj_area < 1.2 and overlap/obj_area > 0.8:
            max_overlap = overlap
            max_name = obj_candidate['name']
            max_bbox = candidate_bbox
    if max_overlap > 0:
        return max_name, max_bbox
    return None, None

def update_memory(memory, mind_name, fluent, loc):
    if fluent == 0 or fluent == 2:
        memory[mind_name]['loc'] = loc
    elif fluent == 1:
        memory[mind_name]['loc'] = None

    return memory

def get_grid_location_using_bbox(obj_frame):
    x_min = obj_frame[0]
    y_min = obj_frame[1]
    x_max = obj_frame[0] + obj_frame[2]
    y_max = obj_frame[1] + obj_frame[3]
    gridLW = 1280 / 25.
    gridLH = 720 / 15.
    center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
    X, Y = int(center_x / gridLW), int(center_y / gridLH)
    return X, Y

def test_data_seq(prj_name):
    if prj_name == 'gru':
        net = MindGRU()
        net.load_state_dict(torch.load('./cptk_gru/model_350.pth'))
        seq_len = 5
    elif prj_name == 'single':
        net = MindHog()
        net.load_state_dict(torch.load('./cptk_single/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined':
        net = MLP_Combined()
        net.load_state_dict(torch.load('./cptk_combined/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined_event':
        net = MLP_Combined()
        net.load_state_dict(torch.load('./cptk_combined_event/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined_att':
        net = MLP_Combined_Att()
        net.load_state_dict(torch.load('./cptk_combined_att/model_best.pth'))
        seq_len = 1
    elif prj_name == 'lstm_sep':
        net = MindLSTMHog()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    elif prj_name == 'event_memory':
        net = MLP_Event_Memory()
        net.load_state_dict(torch.load('./cptk_event_memory/model_best.pth'))
        seq_len = 1
    else:
        net = MindLSTMSep()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_output/'
    event_tree_path = './BestTree_ours_0531_all_clips/'

    attmat = AttMat()
    attmat.load_state_dict(torch.load('../obj_oriented_event/cptk/model_best.pth'))
    attmat.cuda()
    attmat.eval()

    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    for clip in mind_test_clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        with open(feature_path + clip, 'rb') as f:
            features = pickle.load(f)
        with open(event_tree_path + clip, 'rb') as f:
            event_tree = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events_tree_search(event_tree, len(img_names))
        # battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
            p1_hog = features[1]
            p2_hog = features[2]
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
            p1_hog = features[2]
            p2_hog = features[1]
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(p1_events_by_frame.shape[0]):

                event_input = np.zeros((seq_len, 19))
                obj_patch_input = np.zeros((seq_len, 3, 224, 224))
                hog_input = np.zeros((seq_len, 162*2))
                obj_record = obj_records[obj_name][frame_id]

                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                for i in range(-(seq_len - 1), 1, 1):
                    curr_frame_id = max(frame_id + i, 0)
                    # curr_loc
                    curr_df = annt.loc[(annt.frame == curr_frame_id) & (annt.name == obj_name)]
                    curr_loc = get_grid_location(curr_df)
                    # event
                    # hog
                    hog_tracker = p1_hog[frame_id][-162-10:-10]
                    hog_battery = p2_hog[frame_id][-162-10:-10]
                    hog_feature = np.hstack([hog_tracker, hog_battery])
                    hog_input[i + seq_len - 1, :] = hog_feature
                    # obj_patch
                    img = cv2.imread(img_names[frame_id])
                    obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                    x_min = obj_frame['x_min'].item()
                    y_min = obj_frame['y_min'].item()
                    x_max = obj_frame['x_max'].item()
                    y_max = obj_frame['y_max'].item()
                    img_patch = img[y_min:y_max, x_min:x_max]
                    obj_patch = transforms.ToPILImage()(img_patch)
                    obj_patch = obj_transforms(obj_patch).numpy()
                    obj_patch_input[i + seq_len - 1, ...] = obj_patch
                    hog_input = torch.from_numpy(hog_input).float().cuda().view((1, -1))

                    obj_patch = torch.FloatTensor(obj_patch_input).cuda()
                    obj_patch = obj_patch.view((1, 3, 224, 224))
                    att_output = attmat(obj_patch, hog_input)
                    att_output = att_output.data.cpu().numpy().reshape(-1)
                    # output_binary = (att_output > 0.5).astype(float)
                    #
                    # if output_binary[1] == 1 or output_binary[2] == 1:
                    #     p1_event = p1_events_by_frame[curr_frame_id]
                    #     p2_event = p2_events_by_frame[curr_frame_id]
                    # else:
                    #     p1_event = np.array([0., 0., 0.])
                    #     p2_event = np.array([0., 0., 0.])
                    # p1_event = np.exp(p1_event)/np.exp(p1_event).sum()
                    # p2_event = np.exp(p2_event)/np.exp(p2_event).sum()
                    p1_event = p1_events_by_frame[curr_frame_id]
                    p2_event = p2_events_by_frame[curr_frame_id]
                    event_input[i + seq_len - 1, :3] = p1_event
                    event_input[i + seq_len - 1, 3:6] = p2_event
                    # memory
                    memory_dist = []
                    indicator = []
                    for mind_name in memory.keys():
                        if mind_name == 'mg':
                            continue
                        if curr_frame_id == 0:
                            memory_dist.append(0)
                            indicator.append(0)
                        else:
                            if frame_id%50 == 0:
                                memory_loc = obj_records[obj_name][curr_frame_id - 1][mind_name]['loc']
                            else:
                                memory_loc = memory[mind_name]['loc']
                            # memory_loc = obj_records[obj_name][curr_frame_id - 1][mind_name]['loc']
                            if memory_loc is not None:
                                curr_loc = np.array(curr_loc)
                                memory_loc = np.array(memory_loc)
                                memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                                indicator.append(1)
                            else:
                                memory_dist.append(0)
                                indicator.append(0)
                    # get predicted value
                    memory_dist = np.array(memory_dist)
                    indicator = np.array(indicator)
                    event_input[i + seq_len - 1, 6:6 + 5] = memory_dist
                    event_input[i + seq_len - 1, 6+5: 6+5+5] = indicator

                    event_input[i + seq_len - 1, 6+5+5:] = att_output

                # get input
                event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))

                m = net(event_input)
                mind_pred = torch.softmax(m, dim=-1).data.cpu().numpy()
                max_score, idx_m = torch.max(m, 1)
                idx_m = idx_m.cpu().numpy()[0]
                pred_combination = mind_combination[idx_m]

                mc_predict.append(pred_combination[4])
                m1_predict.append(pred_combination[0])
                m2_predict.append(pred_combination[1])
                m12_predict.append(pred_combination[2])
                m21_predict.append(pred_combination[3])
                memory = update_memory(memory, 'm1', pred_combination[0], curr_loc)
                memory = update_memory(memory, 'm2', pred_combination[1], curr_loc)
                memory = update_memory(memory, 'm12', pred_combination[2], curr_loc)
                memory = update_memory(memory, 'm21', pred_combination[3], curr_loc)
                memory = update_memory(memory, 'mc', pred_combination[4], curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

    # with open('./cptk_' + prj_name + '/' + 'output.p', 'wb') as f:
    #     pickle.dump({'mc':[mc_predict, mc_real], 'm1':[m1_predict, m1_real], 'm2':[m2_predict, m2_real],
    #                 'm12':[m12_predict, m12_real], 'm21':[m21_predict, m21_real]}, f)
    results_mc = metrics.classification_report(mc_real, mc_predict, digits=3)
    results_m1 = metrics.classification_report(m1_real, m1_predict, digits=3)
    results_m2 = metrics.classification_report(m2_real, m2_predict, digits=3)
    results_m12 = metrics.classification_report(m12_real, m12_predict, digits=3)
    results_m21 = metrics.classification_report(m21_real, m21_predict, digits=3)
    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    cmc = metrics.confusion_matrix(mc_real, mc_predict)
    cm1 = metrics.confusion_matrix(m1_real, m1_predict)
    cm2 = metrics.confusion_matrix(m2_real, m2_predict)
    cm12 = metrics.confusion_matrix(m12_real, m12_predict)
    cm21 = metrics.confusion_matrix(m21_real, m21_predict)
    plot_confusion_matrix(cmc)
    plot_confusion_matrix(cm1)
    plot_confusion_matrix(cm2)
    plot_confusion_matrix(cm12)
    plot_confusion_matrix(cm21)
    score1 = metrics.f1_score(mc_real, mc_predict, average='macro')
    score2 = metrics.f1_score(m1_real, m1_predict, average='macro')
    score3 = metrics.f1_score(m2_real, m2_predict, average='macro')
    score4 = metrics.f1_score(m12_real, m12_predict, average='macro')
    score5 = metrics.f1_score(m21_real, m21_predict, average='macro')
    print([score1, score2, score3, score4, score5])

    with open('./cptk_' + prj_name + '/' + 'seq_att_remove_event.p', 'wb') as f:
        pickle.dump([[results_m1, results_m2, results_m12, results_m21, results_mc],
                    [score1, score2, score3, score4, score5],
                     [cmc, cm1, cm2, cm12, cm21]], f)

def test_data_seq_per_clip(prj_name):
    if prj_name == 'gru':
        net = MindGRU()
        net.load_state_dict(torch.load('./cptk_gru/model_350.pth'))
        seq_len = 5
    elif prj_name == 'single':
        net = MindHog()
        net.load_state_dict(torch.load('./cptk_single/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined':
        net = MLP_Combined()
        net.load_state_dict(torch.load('./cptk_combined/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined_event':
        net = MLP_Combined()
        net.load_state_dict(torch.load('./cptk_combined_event/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined_att':
        net = MLP_Combined_Att()
        net.load_state_dict(torch.load('./cptk_combined_att/model_best.pth'))
        seq_len = 1
    elif prj_name == 'lstm_sep':
        net = MindLSTMHog()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    elif prj_name == 'event_memory':
        net = MLP_Event_Memory()
        net.load_state_dict(torch.load('./cptk_event_memory/model_best.pth'))
        seq_len = 1
    else:
        net = MindLSTMSep()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_output/'
    event_tree_path = './BestTree_ours_0531_all_clips/'

    attmat = AttMat()
    attmat.load_state_dict(torch.load('../obj_oriented_event/cptk/model_best.pth'))
    attmat.cuda()
    attmat.eval()

    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    clips = os.listdir(event_label_path)

    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    score_total = []
    for clip in mind_test_clips:
        m1_predict, m1_real = [], []
        m2_predict, m2_real = [], []
        m12_predict, m12_real = [], []
        m21_predict, m21_real = [], []
        mc_predict, mc_real = [], []
        mg_predict, mg_real = [], []
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        with open(feature_path + clip, 'rb') as f:
            features = pickle.load(f)
        with open(event_tree_path + clip, 'rb') as f:
            event_tree = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events_tree_search(event_tree, len(img_names))
        # battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
            p1_hog = features[1]
            p2_hog = features[2]
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
            p1_hog = features[2]
            p2_hog = features[1]
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(p1_events_by_frame.shape[0]):

                event_input = np.zeros((seq_len, 19))
                obj_patch_input = np.zeros((seq_len, 3, 224, 224))
                hog_input = np.zeros((seq_len, 162*2))
                obj_record = obj_records[obj_name][frame_id]

                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                for i in range(-(seq_len - 1), 1, 1):
                    curr_frame_id = max(frame_id + i, 0)
                    # curr_loc
                    curr_df = annt.loc[(annt.frame == curr_frame_id) & (annt.name == obj_name)]
                    curr_loc = get_grid_location(curr_df)
                    # event
                    # hog
                    hog_tracker = p1_hog[frame_id][-162-10:-10]
                    hog_battery = p2_hog[frame_id][-162-10:-10]
                    hog_feature = np.hstack([hog_tracker, hog_battery])
                    hog_input[i + seq_len - 1, :] = hog_feature
                    # obj_patch
                    img = cv2.imread(img_names[frame_id])
                    obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                    x_min = obj_frame['x_min'].item()
                    y_min = obj_frame['y_min'].item()
                    x_max = obj_frame['x_max'].item()
                    y_max = obj_frame['y_max'].item()
                    img_patch = img[y_min:y_max, x_min:x_max]
                    obj_patch = transforms.ToPILImage()(img_patch)
                    obj_patch = obj_transforms(obj_patch).numpy()
                    obj_patch_input[i + seq_len - 1, ...] = obj_patch
                    hog_input = torch.from_numpy(hog_input).float().cuda().view((1, -1))

                    obj_patch = torch.FloatTensor(obj_patch_input).cuda()
                    obj_patch = obj_patch.view((1, 3, 224, 224))
                    att_output = attmat(obj_patch, hog_input)
                    att_output = att_output.data.cpu().numpy().reshape(-1)
                    output_binary = (att_output > 0.5).astype(float)

                    if output_binary[1] == 1 or output_binary[2] == 1:
                        p1_event = p1_events_by_frame[curr_frame_id]
                        p2_event = p2_events_by_frame[curr_frame_id]
                    else:
                        p1_event = np.array([0., 0., 0.])
                        p2_event = np.array([0., 0., 0.])
                    # p1_event = np.exp(p1_event)/np.exp(p1_event).sum()
                    # p2_event = np.exp(p2_event)/np.exp(p2_event).sum()
                    # p1_event = p1_events_by_frame[curr_frame_id]
                    # p2_event = p2_events_by_frame[curr_frame_id]
                    event_input[i + seq_len - 1, :3] = p1_event
                    event_input[i + seq_len - 1, 3:6] = p2_event
                    # memory
                    memory_dist = []
                    indicator = []
                    for mind_name in memory.keys():
                        if mind_name == 'mg':
                            continue
                        if curr_frame_id == 0:
                            memory_dist.append(0)
                            indicator.append(0)
                        else:
                            if frame_id%50 == 0:
                                memory_loc = obj_records[obj_name][curr_frame_id - 1][mind_name]['loc']
                            else:
                                memory_loc = memory[mind_name]['loc']
                            # memory_loc = obj_records[obj_name][curr_frame_id - 1][mind_name]['loc']
                            if memory_loc is not None:
                                curr_loc = np.array(curr_loc)
                                memory_loc = np.array(memory_loc)
                                memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                                indicator.append(1)
                            else:
                                memory_dist.append(0)
                                indicator.append(0)
                    # get predicted value
                    memory_dist = np.array(memory_dist)
                    indicator = np.array(indicator)
                    event_input[i + seq_len - 1, 6:6 + 5] = memory_dist
                    event_input[i + seq_len - 1, 6+5: 6+5+5] = indicator

                    event_input[i + seq_len - 1, 6+5+5:] = att_output

                # get input
                event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))

                m = net(event_input)
                mind_pred = torch.softmax(m, dim=-1).data.cpu().numpy()
                max_score, idx_m = torch.max(m, 1)
                idx_m = idx_m.cpu().numpy()[0]
                pred_combination = mind_combination[idx_m]

                mc_predict.append(pred_combination[4])
                m1_predict.append(pred_combination[0])
                m2_predict.append(pred_combination[1])
                m12_predict.append(pred_combination[2])
                m21_predict.append(pred_combination[3])
                memory = update_memory(memory, 'm1', pred_combination[0], curr_loc)
                memory = update_memory(memory, 'm2', pred_combination[1], curr_loc)
                memory = update_memory(memory, 'm12', pred_combination[2], curr_loc)
                memory = update_memory(memory, 'm21', pred_combination[3], curr_loc)
                memory = update_memory(memory, 'mc', pred_combination[4], curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)
        score1 = metrics.f1_score(mc_real, mc_predict, average='macro')
        score2 = metrics.f1_score(m1_real, m1_predict, average='macro')
        score3 = metrics.f1_score(m2_real, m2_predict, average='macro')
        score4 = metrics.f1_score(m12_real, m12_predict, average='macro')
        score5 = metrics.f1_score(m21_real, m21_predict, average='macro')
        score_total.append(score1 + score2 + score3 + score4 + score5)
        with open('./cptk_f1_score_output_likelihood/' + clip, 'wb') as f:
            pickle.dump([[mc_predict, m1_predict, m2_predict, m12_predict, m21_predict],
                        [mc_real, m1_real, m2_real, m12_real, m21_real]], f)

    assert len(score_total) == len(mind_test_clips)
    score_total = np.array(score_total)
    idx = np.argsort(score_total)[::-1][:5]
    for i in idx:
        print(mind_test_clips[i], score_total[i])

    # with open('./cptk_' + prj_name + '/' + 'seq_att_remove_event.p', 'wb') as f:
    #     pickle.dump([[results_m1, results_m2, results_m12, results_m21, results_mc],
    #                 [score1, score2, score3, score4, score5],
    #                  [cmc, cm1, cm2, cm12, cm21]], f)

def test_data_seq_ablation(prj_name):
    if prj_name == 'gru':
        net = MindGRU()
        net.load_state_dict(torch.load('./cptk_gru/model_350.pth'))
        seq_len = 5
    elif prj_name == 'single':
        net = MindHog()
        net.load_state_dict(torch.load('./cptk_single/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined':
        net = MLP_Combined()
        net.load_state_dict(torch.load('./cptk_combined/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined_event':
        net = MLP_Combined()
        net.load_state_dict(torch.load('./cptk_combined_event/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined_att' or prj_name == 'combined_att_add_1024':
        net = MLP_Combined_Att()
        net.load_state_dict(torch.load('./cptk_combined_att_add_1024/model_best.pth'))
        seq_len = 1
    elif prj_name == 'lstm_sep':
        net = MindLSTMHog()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    elif prj_name == 'event_memory':
        net = MLP_Event_Memory()
        net.load_state_dict(torch.load('./cptk_event_memory/model_best.pth'))
        seq_len = 1
    else:
        net = MindLSTMSep()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_output/'
    # event_tree_path = './BestTree_ours_0531_all_clips/'
    event_tree_path = '../mind_search/BestTree_ours_event_likelihood_0531/'

    attmat = AttMat()
    attmat.load_state_dict(torch.load('../obj_oriented_event/cptk/model_best.pth'))
    attmat.cuda()
    attmat.eval()

    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    for clip in mind_test_clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        with open(feature_path + clip, 'rb') as f:
            features = pickle.load(f)
        with open(event_tree_path + clip, 'rb') as f:
            event_tree = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events_tree_search(event_tree, len(img_names))
        # battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
            p1_hog = features[1]
            p2_hog = features[2]
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
            p1_hog = features[2]
            p2_hog = features[1]
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(p1_events_by_frame.shape[0]):

                event_input = np.zeros((seq_len, 19))
                obj_patch_input = np.zeros((seq_len, 3, 224, 224))
                hog_input = np.zeros((seq_len, 162*2))
                obj_record = obj_records[obj_name][frame_id]

                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                for i in range(-(seq_len - 1), 1, 1):
                    curr_frame_id = max(frame_id + i, 0)
                    # curr_loc
                    curr_df = annt.loc[(annt.frame == curr_frame_id) & (annt.name == obj_name)]
                    curr_loc = get_grid_location(curr_df)
                    # event
                    # hog
                    hog_tracker = p1_hog[frame_id][-162-10:-10]
                    hog_battery = p2_hog[frame_id][-162-10:-10]
                    hog_feature = np.hstack([hog_tracker, hog_battery])
                    hog_input[i + seq_len - 1, :] = hog_feature
                    # obj_patch
                    img = cv2.imread(img_names[frame_id])
                    obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                    x_min = obj_frame['x_min'].item()
                    y_min = obj_frame['y_min'].item()
                    x_max = obj_frame['x_max'].item()
                    y_max = obj_frame['y_max'].item()
                    img_patch = img[y_min:y_max, x_min:x_max]
                    obj_patch = transforms.ToPILImage()(img_patch)
                    obj_patch = obj_transforms(obj_patch).numpy()
                    obj_patch_input[i + seq_len - 1, ...] = obj_patch
                    hog_input = torch.from_numpy(hog_input).float().cuda().view((1, -1))

                    obj_patch = torch.FloatTensor(obj_patch_input).cuda()
                    obj_patch = obj_patch.view((1, 3, 224, 224))
                    att_output = attmat(obj_patch, hog_input)
                    att_output = att_output.data.cpu().numpy().reshape(-1)
                    output_binary = (att_output > 0.5).astype(float)

                    if output_binary[1] == 1 or output_binary[2] == 1:
                        p1_event = p1_events_by_frame[curr_frame_id]
                        p2_event = p2_events_by_frame[curr_frame_id]
                    else:
                        p1_event = np.array([0., 0., 0.])
                        p2_event = np.array([0., 0., 0.])
                    # p1_event = np.array([0., 0., 0.])
                    # p2_event = np.array([0., 0., 0.])
                    # p1_event = np.array([1/3., 1/3., 1/3.])
                    # p2_event = np.array([1/3., 1/3., 1/3.])
                    # p1_event = p1_events_by_frame[curr_frame_id]
                    # p2_event = p2_events_by_frame[curr_frame_id]
                    # p1_event = p1_events_by_frame[curr_frame_id]
                    # p2_event = p2_events_by_frame[curr_frame_id]
                    # p1_event = np.exp(p1_event)/np.exp(p1_event).sum()
                    # p2_event = np.exp(p2_event)/np.exp(p2_event).sum()
                    event_input[i + seq_len - 1, :3] = p1_event
                    event_input[i + seq_len - 1, 3:6] = p2_event
                    # memory
                    memory_dist = []
                    indicator = []
                    for mind_name in memory.keys():
                        if mind_name == 'mg':
                            continue
                        if curr_frame_id == 0:
                            memory_dist.append(0)
                            indicator.append(0)
                        else:
                            if frame_id%50 == 0:
                                memory_loc = obj_records[obj_name][curr_frame_id - 1][mind_name]['loc']
                            else:
                                memory_loc = memory[mind_name]['loc']
                            # memory_loc = obj_records[obj_name][curr_frame_id - 1][mind_name]['loc']
                            # memory_loc = memory[mind_name]['loc']
                            if memory_loc is not None:
                                curr_loc = np.array(curr_loc)
                                memory_loc = np.array(memory_loc)
                                memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                                indicator.append(1)
                            else:
                                memory_dist.append(0)
                                indicator.append(0)
                    # get predicted value
                    memory_dist = np.array(memory_dist)
                    indicator = np.array(indicator)
                    event_input[i + seq_len - 1, 6:6 + 5] = memory_dist
                    event_input[i + seq_len - 1, 6+5: 6+5+5] = indicator

                    event_input[i + seq_len - 1, 6+5+5:] = att_output

                # get input
                event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))

                m = net(event_input)
                mind_pred = torch.softmax(m, dim=-1).data.cpu().numpy()
                max_score, idx_m = torch.max(m, 1)
                idx_m = idx_m.cpu().numpy()[0]
                pred_combination = mind_combination[idx_m]

                mc_predict.append(pred_combination[4])
                m1_predict.append(pred_combination[0])
                m2_predict.append(pred_combination[1])
                m12_predict.append(pred_combination[2])
                m21_predict.append(pred_combination[3])
                memory = update_memory(memory, 'm1', pred_combination[0], curr_loc)
                memory = update_memory(memory, 'm2', pred_combination[1], curr_loc)
                memory = update_memory(memory, 'm12', pred_combination[2], curr_loc)
                memory = update_memory(memory, 'm21', pred_combination[3], curr_loc)
                memory = update_memory(memory, 'mc', pred_combination[4], curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

    # with open('./cptk_' + prj_name + '/' + 'output.p', 'wb') as f:
    #     pickle.dump({'mc':[mc_predict, mc_real], 'm1':[m1_predict, m1_real], 'm2':[m2_predict, m2_real],
    #                 'm12':[m12_predict, m12_real], 'm21':[m21_predict, m21_real]}, f)
    results_mc = metrics.classification_report(mc_real, mc_predict, digits=3)
    results_m1 = metrics.classification_report(m1_real, m1_predict, digits=3)
    results_m2 = metrics.classification_report(m2_real, m2_predict, digits=3)
    results_m12 = metrics.classification_report(m12_real, m12_predict, digits=3)
    results_m21 = metrics.classification_report(m21_real, m21_predict, digits=3)
    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    cmc = metrics.confusion_matrix(mc_real, mc_predict)
    cm1 = metrics.confusion_matrix(m1_real, m1_predict)
    cm2 = metrics.confusion_matrix(m2_real, m2_predict)
    cm12 = metrics.confusion_matrix(m12_real, m12_predict)
    cm21 = metrics.confusion_matrix(m21_real, m21_predict)
    plot_confusion_matrix(cmc)
    plot_confusion_matrix(cm1)
    plot_confusion_matrix(cm2)
    plot_confusion_matrix(cm12)
    plot_confusion_matrix(cm21)
    score1 = metrics.f1_score(mc_real, mc_predict, average='macro')
    score2 = metrics.f1_score(m1_real, m1_predict, average='macro')
    score3 = metrics.f1_score(m2_real, m2_predict, average='macro')
    score4 = metrics.f1_score(m12_real, m12_predict, average='macro')
    score5 = metrics.f1_score(m21_real, m21_predict, average='macro')
    print([score1, score2, score3, score4, score5])

    with open('./cptk_' + prj_name + '/' + 'seq_event_likelihood_50.p', 'wb') as f:
        pickle.dump([[results_m1, results_m2, results_m12, results_m21, results_mc],
                    [score1, score2, score3, score4, score5],
                     [cmc, cm1, cm2, cm12, cm21]], f)

def change2vec(predict, mc_real):
    predict = np.array(predict)
    mc_real = np.array(mc_real)

    mc_reall = np.zeros((mc_real.size, mc_real.max() + 1))
    mc_reall[np.arange(mc_real.size), mc_real] = 1

    return predict, mc_reall

def test_data_seq_baseline(prj_name):
    if prj_name == 'gru':
        net = MindGRU()
        net.load_state_dict(torch.load('./cptk_gru/model_350.pth'))
        seq_len = 5
    elif prj_name == 'single':
        net = MindHog()
        net.load_state_dict(torch.load('./cptk_single/model_best.pth'))
        seq_len = 1
    elif prj_name == 'combined_cnn':
        net = MindCNN()
        net.load_state_dict(torch.load('./cptk_combined_cnn/model_best.pth'))
        seq_len = 1
    elif prj_name == 'raw_feature':
        net = MLP_Feature()
        net.load_state_dict(torch.load('./cptk_raw_feature/model_best.pth'))
        seq_len = 1
    elif prj_name == 'lstm_cnn':
        net = MindLSTMHogCNN()
        net.load_state_dict(torch.load('./cptk_lstm_cnn/model_best.pth'))
        seq_len = 5
    elif prj_name == 'lstm_sep':
        net = MindLSTMHog()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    elif prj_name == 'cnn_no_hog':
        net = MindCNNNoHog()
        net.load_state_dict(torch.load('./cptk_cnn_no_hog/model_best.pth'))
        seq_len = 1
    else:
        net = MindLSTMSep()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_baseline_output/'

    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    print(len(mind_test_clips))
    mind_combination = list(product([0, 1, 2, 3], repeat = 5))
    for clip in mind_test_clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(feature_path + clip, 'rb') as f:
            features = pickle.load(f)

        if person_ids[clip.split('.')[0]] == 'P1':
            p1_hog = features[1]
            p2_hog = features[2]
        else:
            p1_hog = features[2]
            p2_hog = features[1]
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(len(p1_hog)):

                obj_patch_input = np.zeros((seq_len, 3, 224, 224))
                hog_input = np.zeros((seq_len, 162*2))
                obj_record = obj_records[obj_name][frame_id]

                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                for i in range(-(seq_len - 1), 1, 1):
                    # hog
                    hog_tracker = p1_hog[frame_id][-162-10:-10]
                    hog_battery = p2_hog[frame_id][-162-10:-10]
                    hog_feature = np.hstack([hog_tracker, hog_battery])
                    hog_input[i + seq_len - 1, :] = hog_feature
                    # obj_patch
                    img = cv2.imread(img_names[frame_id])
                    obj_patch = transforms.ToPILImage()(img)
                    obj_patch = obj_transforms(obj_patch).numpy()
                    obj_patch_input[i + seq_len - 1, ...] = obj_patch


                # get input
                if seq_len > 1:
                    hog_input = torch.from_numpy(hog_input).float().cuda().view((1, seq_len, -1))

                    obj_patch = torch.FloatTensor(obj_patch_input).cuda()
                    obj_patch = obj_patch.view((1, seq_len, 3, 224, 224))
                else:
                    hog_input = torch.from_numpy(hog_input).float().cuda().view((1, -1))

                    obj_patch = torch.FloatTensor(obj_patch_input).cuda()
                    obj_patch = obj_patch.view((1, 3, 224, 224))

                m1 = net(obj_patch)
                max_score, idx_m1 = torch.max(m1, 1)
                pred_combination = mind_combination[idx_m1.data.cpu().numpy()[0]]
                m1_predict.append(pred_combination[0])
                m2_predict.append(pred_combination[1])
                m12_predict.append(pred_combination[2])
                m21_predict.append(pred_combination[3])
                mc_predict.append(pred_combination[4])
                # m1_predict.append(m1.data.cpu().numpy()[0])
                # m2_predict.append(m2.data.cpu().numpy()[0])
                # m12_predict.append(m12.data.cpu().numpy()[0])
                # m21_predict.append(m21.data.cpu().numpy()[0])
                # mc_predict.append(mc.data.cpu().numpy()[0])

    # predict, real = change2vec(mc_predict, mc_real)
    # print(metrics.average_precision_score(real, predict, average = 'weighted'))

    # with open('./cptk_' + prj_name + '/' + 'output_mc.p', 'wb') as f:
    #     pickle.dump([mc_predict, mc_real], f)
    # with open('./cptk_' + prj_name + '/' + 'output_m1.p', 'wb') as f:
    #     pickle.dump([m1_predict, m1_real], f)
    # with open('./cptk_' + prj_name + '/' + 'output_m2.p', 'wb') as f:
    #     pickle.dump([m2_predict, m2_real], f)
    # with open('./cptk_' + prj_name + '/' + 'output_m12.p', 'wb') as f:
    #     pickle.dump([m12_predict, m12_real], f)
    # with open('./cptk_' + prj_name + '/' + 'output_m21.p', 'wb') as f:
    #     pickle.dump([m21_predict, m21_real], f)



    results_mc = metrics.classification_report(mc_real, mc_predict, digits=3)
    results_m1 = metrics.classification_report(m1_real, m1_predict, digits=3)
    results_m2 = metrics.classification_report(m2_real, m2_predict, digits=3)
    results_m12 = metrics.classification_report(m12_real, m12_predict, digits=3)
    results_m21 = metrics.classification_report(m21_real, m21_predict, digits=3)
    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    cmc = metrics.confusion_matrix(mc_real, mc_predict)
    cm1 = metrics.confusion_matrix(m1_real, m1_predict)
    cm2 = metrics.confusion_matrix(m2_real, m2_predict)
    cm12 = metrics.confusion_matrix(m12_real, m12_predict)
    cm21 = metrics.confusion_matrix(m21_real, m21_predict)
    plot_confusion_matrix(cmc)
    plot_confusion_matrix(cm1)
    plot_confusion_matrix(cm2)
    plot_confusion_matrix(cm12)
    plot_confusion_matrix(cm21)
    score1 = metrics.f1_score(mc_real, mc_predict, average='macro')
    score2 = metrics.f1_score(m1_real, m1_predict, average='macro')
    score3 = metrics.f1_score(m2_real, m2_predict, average='macro')
    score4 = metrics.f1_score(m12_real, m12_predict, average='macro')
    score5 = metrics.f1_score(m21_real, m21_predict, average='macro')
    print([score1, score2, score3, score4, score5])

    with open('./cptk_' + prj_name + '/' + 'seq.p', 'wb') as f:
        pickle.dump([[results_m1, results_m2, results_m12, results_m21, results_mc],
                     [score1, score2, score3, score4, score5], [cmc, cm1, cm2, cm12, cm21]], f)

def test_data_seq_random(prj_name):

    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    save_path = '/home/shuwen/data/data_preprocessing2/mind_baseline_output/'

    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    for clip in mind_test_clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)

        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        frames = annt.frame.unique()
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(len(frames)):
                obj_record = obj_records[obj_name][frame_id]
                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                m1_predict.append(np.random.choice(4, 1))
                m2_predict.append(np.random.choice(4, 1))
                m12_predict.append(np.random.choice(4, 1))
                m21_predict.append(np.random.choice(4, 1))
                mc_predict.append(np.random.choice(4, 1))

    with open('./cptk_' + prj_name + '/' + 'output.p', 'wb') as f:
        pickle.dump({'mc':[mc_predict, mc_real], 'm1':[m1_predict, m1_real], 'm2':[m2_predict, m2_real],
                    'm12':[m12_predict, m12_real], 'm21':[m21_predict, m21_real]}, f)
    results_mc = metrics.classification_report(mc_real, mc_predict, digits=3)
    results_m1 = metrics.classification_report(m1_real, m1_predict, digits=3)
    results_m2 = metrics.classification_report(m2_real, m2_predict, digits=3)
    results_m12 = metrics.classification_report(m12_real, m12_predict, digits=3)
    results_m21 = metrics.classification_report(m21_real, m21_predict, digits=3)
    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    cmc = metrics.confusion_matrix(mc_real, mc_predict)
    cm1 = metrics.confusion_matrix(m1_real, m1_predict)
    cm2 = metrics.confusion_matrix(m2_real, m2_predict)
    cm12 = metrics.confusion_matrix(m12_real, m12_predict)
    cm21 = metrics.confusion_matrix(m21_real, m21_predict)
    plot_confusion_matrix(cmc)
    plot_confusion_matrix(cm1)
    plot_confusion_matrix(cm2)
    plot_confusion_matrix(cm12)
    plot_confusion_matrix(cm21)
    score1 = metrics.accuracy_score(mc_real, mc_predict)
    score2 = metrics.accuracy_score(m1_real, m1_predict)
    score3 = metrics.accuracy_score(m2_real, m2_predict)
    score4 = metrics.accuracy_score(m12_real, m12_predict)
    score5 = metrics.accuracy_score(m21_real, m21_predict)
    print([score1, score2, score3, score4, score5])

    with open('./cptk_' + prj_name + '/' + 'seq.p', 'wb') as f:
        pickle.dump([[results_m1, results_m2, results_m12, results_m21, results_mc], [score1, score2, score3, score4, score5], [cmc, cm1, cm2, cm12, cm21]], f)

def test_data_seq_tree_search(prj_name):
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'

    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    print(len(mind_test_clips))
    for clip in mind_test_clips[:1]:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)

        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        frames = annt.frame.unique()
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            with open('../mind_search/mind_posterior_ours_combined_0531/' + clip.split('.')[0] + '/' + obj_name + '.p', 'rb') as f:
                search_results = pickle.load(f)

            for frame_id in range(len(frames)):

                obj_record = obj_records[obj_name][frame_id]

                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])


                combination = search_results[frame_id]['mind']
                m1_predict.append(combination[0])
                m2_predict.append(combination[1])
                m12_predict.append(combination[2])
                m21_predict.append(combination[3])
                mc_predict.append(combination[4])

    results_mc = metrics.classification_report(mc_real, mc_predict, digits=3)
    results_m1 = metrics.classification_report(m1_real, m1_predict, digits=3)
    results_m2 = metrics.classification_report(m2_real, m2_predict, digits=3)
    results_m12 = metrics.classification_report(m12_real, m12_predict, digits=3)
    results_m21 = metrics.classification_report(m21_real, m21_predict, digits=3)
    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    cmc = metrics.confusion_matrix(mc_real, mc_predict)
    cm1 = metrics.confusion_matrix(m1_real, m1_predict)
    cm2 = metrics.confusion_matrix(m2_real, m2_predict)
    cm12 = metrics.confusion_matrix(m12_real, m12_predict)
    cm21 = metrics.confusion_matrix(m21_real, m21_predict)
    plot_confusion_matrix(cmc)
    plot_confusion_matrix(cm1)
    plot_confusion_matrix(cm2)
    plot_confusion_matrix(cm12)
    plot_confusion_matrix(cm21)
    score1 = metrics.accuracy_score(mc_real, mc_predict)
    score2 = metrics.accuracy_score(m1_real, m1_predict)
    score3 = metrics.accuracy_score(m2_real, m2_predict)
    score4 = metrics.accuracy_score(m12_real, m12_predict)
    score5 = metrics.accuracy_score(m21_real, m21_predict)
    print([score1, score2, score3, score4, score5])

    with open('./cptk_' + prj_name + '/' + 'seq_combined.p', 'wb') as f:
        pickle.dump([[results_m1, results_m2, results_m12, results_m21, results_mc],
                     [score1, score2, score3, score4, score5], [cmc, cm1, cm2, cm12, cm21]], f)

def test_data_seq_tree_search_per_clip(prj_name):
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'

    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    print(len(mind_test_clips))
    total_score = []
    mind_combination = list(product([0, 1, 2, 3], repeat=5))
    for clip in mind_test_clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)

        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        frames = annt.frame.unique()
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            with open('../mind_search/mind_posterior_ours_combined_0531/' + clip.split('.')[0] + '/' + obj_name + '.p', 'rb') as f:
                search_results = pickle.load(f)

            for frame_id in range(len(frames)):

                obj_record = obj_records[obj_name][frame_id]

                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])


                combination = search_results[frame_id]['mind']
                pre_idx = np.argmax(combination)
                combination = mind_combination[pre_idx]
                m1_predict.append(combination[0])
                m2_predict.append(combination[1])
                m12_predict.append(combination[2])
                m21_predict.append(combination[3])
                mc_predict.append(combination[4])

        score1 = metrics.f1_score(mc_real, mc_predict, average='macro')
        score2 = metrics.f1_score(m1_real, m1_predict, average='macro')
        score3 = metrics.f1_score(m2_real, m2_predict, average='macro')
        score4 = metrics.f1_score(m12_real, m12_predict, average='macro')
        score5 = metrics.f1_score(m21_real, m21_predict, average='macro')
        total_score.append(score1 + score2 + score3 + score4 + score5)

        with open('./cptk_f1_score_output_search/' + clip, 'wb') as f:
            pickle.dump([[mc_predict, m1_predict, m2_predict, m12_predict, m21_predict],
                         [mc_real, m1_real, m2_real, m12_real, m21_real]], f)

    total_score = np.array(total_score)
    idx = np.argsort(total_score)[::-1][:5]
    for i in idx:
        print(mind_test_clips[i], total_score[i])


if __name__ == '__main__':
    main()









