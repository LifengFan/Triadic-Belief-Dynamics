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
from mind_model_combined import *
import seaborn as sn
import random
from torchvision import transforms
import cv2
from itertools import product


# mind_test_clips = ['test_94342_16.p', 'test_boelter4_5.p', 'test_94342_2.p', 'test_boelter4_10.p', 'test_boelter2_3.p', 'test_94342_20.p', 'test_boelter4_9.p', 'test_boelter3_9.p', 'test_boelter3_4.p', 'test_boelter2_12.p', 'test_boelter4_6.p', 'test2.p', 'test_boelter4_2.p', 'test_boelter4_3.p', 'test_94342_24.p', 'test_94342_17.p', 'test_94342_6.p', 'test_94342_8.p', 'test_boelter3_0.p', 'test_94342_11.p', 'test_boelter3_7.p', 'test7.p', 'test_94342_18.p', 'test_boelter4_12.p', 'test_boelter_10.p', 'test_boelter3_8.p', 'test_boelter2_6.p', 'test_boelter4_7.p', 'test_boelter4_8.p', 'test_boelter_12.p', 'test_boelter4_0.p', 'test_boelter2_17.p', 'test_boelter3_12.p', 'test_boelter3_11.p', 'test_boelter3_5.p', 'test_94342_4.p', 'test_94342_15.p', 'test_94342_19.p', 'test_94342_7.p', 'test_boelter2_16.p', 'test_boelter2_8.p', 'test_94342_3.p', 'test_boelter_3.p', 'test_9434_3.p', 'test_boelter2_0.p', 'test_boelter3_13.p', 'test_9434_18.p', 'test_boelter_18.p', 'test_94342_22.p', 'test_boelter_6.p', 'test_boelter_4.p', 'test_boelter3_1.p', 'test_boelter3_2.p', 'test_boelter_7.p', 'test_boelter_13.p', 'test1.p', 'test_boelter3_3.p', 'test_boelter4_11.p', 'test_94342_1.p', 'test_94342_25.p', 'test_boelter_1.p', 'test_boelter_21.p', 'test_boelter3_6.p', 'test_boelter_14.p', 'test_94342_12.p', 'test_boelter2_14.p', 'test_boelter4_13.p', 'test_94342_10.p', 'test_boelter_9.p', 'test_94342_5.p', 'test_boelter_17.p', 'test6.p', 'test_boelter4_4.p', 'test_94342_23.p', 'test_boelter3_10.p', 'test_94342_21.p', 'test_94342_0.p', 'test_boelter_2.p', 'test_9434_1.p', 'test_boelter2_15.p', 'test_boelter4_1.p', 'test_boelter_5.p', 'test_94342_13.p', 'test_94342_14.p', 'test_boelter2_7.p', 'test_boelter_19.p', 'test_boelter_15.p', 'test_94342_26.p']
mind_test_clips = ['test_boelter4_5.p', 'test_94342_2.p', 'test_boelter4_10.p', 'test_boelter2_3.p', 'test_94342_20.p', 'test_boelter3_9.p', 'test_boelter4_6.p', 'test2.p', 'test_boelter4_2.p', 'test_94342_24.p', 'test_94342_17.p', 'test_94342_8.p', 'test_94342_11.p', 'test_boelter3_7.p', 'test_94342_18.p', 'test_boelter_10.p', 'test_boelter3_8.p', 'test_boelter2_6.p', 'test_boelter4_7.p', 'test_boelter4_8.p', 'test_boelter4_0.p', 'test_boelter2_17.p', 'test_boelter3_12.p', 'test_boelter3_5.p', 'test_94342_4.p', 'test_94342_15.p']
def plot_confusion_matrix(cmc):
    df_cm = pd.DataFrame(cmc, range(cmc.shape[0]), range(cmc.shape[1]))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

def get_data(data_path, model_type):

    clips = os.listdir(data_path)
    data = []
    labels = []

    for clip in clips:
        print(clip)
        with open(data_path + clip, 'rb') as f:
            vec_input, label_ = pickle.load(f)
            data = data + vec_input
            labels = labels + label_

    c = list(zip(data, labels))
    random.shuffle(c)
    train_ratio = int(len(c) * 0.6)
    validate_ratio = int(len(c) * 0.2)
    data, label = zip(*c)

    train_x, train_y = data[:train_ratio], label[:train_ratio]
    validate_x, validate_y = data[train_ratio:train_ratio + validate_ratio], label[train_ratio:train_ratio + validate_ratio]
    test_x, test_y = data[train_ratio + validate_ratio:], label[train_ratio + validate_ratio:]
    print(len(train_y), len(validate_y), len(test_y))
    mind_count = np.zeros(1024)

    for data_id in range(len(train_y)):
        mind_count[train_y[data_id]] += 1
    return train_x, train_y, validate_x, validate_y, test_x, test_y, mind_count

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
        self.seq_size = 10
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '../../annotations/'
        self.person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../3d_pose2gaze/record_bbox/'
        self.obj_bbox = '/home/shuwen/data/data_preprocessing2/post_neighbor_smooth_newseq/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'

    def __getitem__(self, index):
        hog_input, img_input = self.train_x[index]
        obj_patch_input = np.zeros((self.seq_size, 3, 224, 224))


        for i in range(len(img_input)):

            # obj_patch
            img_name = self.img_path + '/'.join(img_input[i].split('/')[-3:])
            assert os.path.exists(img_name)
            img = cv2.imread(img_name)
            obj_patch = transforms.ToPILImage()(img)
            obj_patch = self.transforms(obj_patch).numpy()
            obj_patch_input[i, ...] = obj_patch

        actual_val = self.train_y[index]

        return obj_patch_input, hog_input, actual_val

    def __len__(self):
        return len(self.train_x)

def collate_fn_lstm_cnn(batch):
    N = len(batch)
    seq_len = 10

    obj_batch = np.zeros((N, seq_len, 3, 224, 224))
    hog_batch = np.zeros((N, seq_len, 162*2))
    mc_label_batch = np.zeros(N)


    for i, (obj, hog, mc) in enumerate(batch):
        obj_batch[i, ...] = obj
        hog_batch[i, ...] = hog
        mc_label_batch[i,...] = mc


    obj_batch = torch.FloatTensor(obj_batch)
    hog_batch = torch.FloatTensor(hog_batch)
    mc_label_batch = torch.LongTensor(mc_label_batch)

    return obj_batch, hog_batch, mc_label_batch

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

    # data_path = '/home/shuwen/data/data_preprocessing2/mind_lstm_training_add_hog_sep/'
    data_path = '../../data/mind_lstm_training_cnn_att/'
    model_type = 'gru_cnn'

    train_x, train_y, validate_x, validate_y, test_x, test_y, mind_count = get_data(data_path, model_type)

    learningRate = 0.01
    epochs = 300
    batch_size = 64

    train(model_type, learningRate, epochs, batch_size, train_x, train_y, validate_x, validate_y, mind_count) #, checkpoint='./cptk_single/model_best.pth')
    # net = MindHog()
    # net.load_state_dict(torch.load('./cptk_single/model_best.pth'))
    # if torch.cuda.is_available():
    #     net.cuda()
    # net.eval()
    # test_score(net, test_x, test_y, batch_size, model_type, 'test')

    # test_data_seq('single')
    # test_data_seq_baseline('lstm_cnn')


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

    if proj_name == 'single' or proj_name == 'single_one_hot':
        train_set = mydataset(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size,
                                                   shuffle=False)
    elif proj_name == 'cnn':
        train_set = mydataset_cnn(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_cnn, batch_size=batch_size,
                                                   shuffle=False)
    elif proj_name == 'lstm_cnn' or proj_name == 'gru_cnn':
        train_set = mydataset_lstm_cnn(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_lstm_cnn, batch_size=batch_size,
                                                   shuffle=False)
    else:
        train_set = mydataset_lstm(data, label)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_lstm, batch_size=batch_size,
                                                   shuffle=False)


    net.eval()

    mind_combination = list(product([0, 1, 2, 3], repeat=5))
    pbar = tqdm(train_loader)
    for batch in pbar:
        obj_patch, hog_batch, label_batch = batch
        obj_patch = obj_patch.cuda()
        hog_batch = hog_batch.cuda()
        label_batch = label_batch.numpy()

        m = net(obj_patch, hog_batch)
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

        score1 = metrics.accuracy_score(total_act_mc, total_mc)
        score2 = metrics.accuracy_score(total_act_m1, total_m1)
        score3 = metrics.accuracy_score(total_act_m2, total_m2)
        score4 = metrics.accuracy_score(total_act_m12, total_m12)
        score5 = metrics.accuracy_score(total_act_m21, total_m21)
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


def train(save_prefix, learningRate, epochs, batch_size, train_x, train_y, validate_x, validate_y, mind_count, checkpoint = None, startepoch = None):
    if save_prefix == 'single':
        model = MindHog()
    elif save_prefix == 'single_one_hot':
        model = MindHog()
    elif save_prefix == 'cnn':
        model = MindCNN()
    elif save_prefix == 'lstm_cnn' or save_prefix == 'gru_cnn':
        model = MindLSTMHogCNN()
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
    # weights = [1/39., 1/3., 1/391., 1/3702.]
    # weights = torch.FloatTensor(weights).cuda()
    # criterionc = torch.nn.CrossEntropyLoss(weight=weights)
    # weights = [1/87., 1/11., 1/929., 1/3108.]
    # weights = torch.FloatTensor(weights).cuda()
    # criterionm12 = torch.nn.CrossEntropyLoss(weight=weights)
    # weights = [1/78., 1/11., 1/847., 1/3199.]
    # weights = torch.FloatTensor(weights).cuda()
    # criterionm21 = torch.nn.CrossEntropyLoss(weight=weights)
    # weights = [1/175., 1/6., 1/391., 1/3702.]
    # weights = torch.FloatTensor(weights).cuda()
    # criterionm1 = torch.nn.CrossEntropyLoss(weight=weights)
    # weights = [1/177., 1/5., 1/2599., 1/1354.]
    # weights = torch.FloatTensor(weights).cuda()
    # criterionm2 = torch.nn.CrossEntropyLoss(weight=weights)
    # criterionmg = torch.nn.CrossEntropyLoss()

    mind_count[mind_count > 0] = 1. / mind_count[mind_count > 0]
    weights = torch.tensor(mind_count).float().cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # criterionmg = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # scheduler = MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.1)
    losses_m1, losses_m2, losses_m12, losses_m21, losses_mc = [], [], [], [], []
    best_score = 0

    if save_prefix == 'single' or save_prefix == 'single_one_hot':
        train_set = mydataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size,
                                                   shuffle=False)
    elif save_prefix == 'cnn':
        train_set = mydataset_cnn(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_cnn, batch_size=batch_size,
                                                   shuffle=False)
    elif save_prefix == 'lstm_cnn' or save_prefix == 'gru_cnn':
        train_set = mydataset_lstm_cnn(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_lstm_cnn, batch_size=batch_size,
                                                   shuffle=False)
    else:
        train_set = mydataset_lstm(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_lstm, batch_size=batch_size,
                                                   shuffle=False)
    losses = []
    for epoch in range(startepoch, epochs):
        model.train()
        # training set -- perform model training
        epoch_training_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            obj_patch, hog_batch, label_batch = batch
            obj_patch = obj_patch.cuda()
            hog_batch = hog_batch.cuda()
            label_batch = label_batch.cuda()

            # Make gradients zero for parameters 'W', 'b'
            optimizer.zero_grad()

            # m1, m2, m12, m21, mc = model(obj_batch)
            label = model(obj_patch, hog_batch)
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
        if epoch%10 == 0: # and epoch > 0 :
            for param in optimizer.param_groups:
                if param['lr'] > 1e-5:
                    param['lr'] = param['lr']*0.5

        if epoch%50 == 0:
            save_path = './cptk_' + save_prefix + '/model_' +str(epoch)+'.pth'
            torch.save(model.state_dict(), save_path)
            plt.plot(losses)
            plt.show()

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
    elif prj_name == 'lstm_sep':
        net = MindLSTMHog()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
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
    for clip in clips[:10]:
        if not os.path.exists(reannotation_path + clip):
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        with open(feature_path + clip, 'rb') as f:
            features = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
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
                flag = 0
                # for i in range(-(seq_len - 1), 1, 1):
                #     curr_frame_id = max(frame_id + i, 0)
                #     p1_event = p1_events_by_frame[curr_frame_id]
                #     p2_event = p2_events_by_frame[curr_frame_id]
                #
                #     if np.all(p1_event == np.array([0, 0, 0])) or np.all(p2_event == np.array([0, 0, 0])):
                #         flag = 1
                # if flag:
                #     m1_predict.append(3)
                #     m2_predict.append(3)
                #     m12_predict.append(3)
                #     m21_predict.append(3)
                #     mc_predict.append(3)
                #     curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                #     curr_loc = get_grid_location(curr_df)
                #     memory = update_memory(memory, 'm1', 3, curr_loc)
                #     memory = update_memory(memory, 'm2', 3, curr_loc)
                #     memory = update_memory(memory, 'm12', 3, curr_loc)
                #     memory = update_memory(memory, 'm21', 3, curr_loc)
                #     memory = update_memory(memory, 'mc', 3, curr_loc)
                #     memory = update_memory(memory, 'mg', 2, curr_loc)
                #     obj_record = obj_records[obj_name][frame_id]
                #     for mind_name in obj_record:
                #         if mind_name == 'm1':
                #             m1_real.append(obj_record[mind_name]['fluent'])
                #         elif mind_name == 'm2':
                #             m2_real.append(obj_record[mind_name]['fluent'])
                #         elif mind_name == 'm12':
                #             m12_real.append(obj_record[mind_name]['fluent'])
                #         elif mind_name == 'm21':
                #             m21_real.append(obj_record[mind_name]['fluent'])
                #         elif mind_name == 'mc':
                #             mc_real.append(obj_record[mind_name]['fluent'])
                #         elif mind_name == 'mg':
                #             mg_real.append(obj_record[mind_name]['fluent'])
                #     continue

                event_input = np.zeros((seq_len, 16))
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
                    p1_event = p1_events_by_frame[curr_frame_id]
                    p2_event = p2_events_by_frame[curr_frame_id]
                    p1_event = np.exp(p1_event)/np.exp(p1_event).sum()
                    p2_event = np.exp(p2_event)/np.exp(p2_event).sum()
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
                    event_input[i + seq_len - 1, 6+5:] = indicator
                    # hog
                    hog_tracker = features[1][frame_id][-162-10:-10]
                    hog_battery = features[2][frame_id][-162-10:-10]
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


                # get input
                event_input = torch.from_numpy(event_input).float().cuda().view((1, seq_len, -1))
                hog_input = torch.from_numpy(hog_input).float().cuda().view((1, seq_len, -1))

                obj_patch = torch.FloatTensor(obj_patch_input).cuda()
                obj_patch = obj_patch.view((1, seq_len, 3, 224, 224))
                m1, m2, m12, m21, mc = net(event_input, obj_patch, hog_input)
                max_score, idx_mc = torch.max(mc, 1)
                max_score, idx_m21 = torch.max(m21, 1)
                max_score, idx_m12 = torch.max(m12, 1)
                max_score, idx_m1 = torch.max(m1, 1)
                max_score, idx_m2 = torch.max(m2, 1)
                m1_predict.append(idx_m1.cpu().numpy()[0])
                m2_predict.append(idx_m2.cpu().numpy()[0])
                m12_predict.append(idx_m12.cpu().numpy()[0])
                m21_predict.append(idx_m21.cpu().numpy()[0])
                mc_predict.append(idx_mc.cpu().numpy()[0])
                # update memory
                curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                curr_loc = get_grid_location(curr_df)
                memory = update_memory(memory, 'm1', idx_m1, curr_loc)
                memory = update_memory(memory, 'm2', idx_m2, curr_loc)
                memory = update_memory(memory, 'm12', idx_m12, curr_loc)
                memory = update_memory(memory, 'm21', idx_m21, curr_loc)
                memory = update_memory(memory, 'mc', idx_mc, curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)
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
    with open('./cptk_' + prj_name + '/' + 'seq.p', 'wb') as f:
        pickle.dump([results_m1, results_m2, results_m12, results_m21, results_mc], f)
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

def test_data_seq_baseline(prj_name):
    if prj_name == 'gru':
        net = MindGRU()
        net.load_state_dict(torch.load('./cptk_gru/model_350.pth'))
        seq_len = 5
    elif prj_name == 'single':
        net = MindHog()
        net.load_state_dict(torch.load('./cptk_single/model_best.pth'))
        seq_len = 1
    elif prj_name == 'cnn':
        net = MindCNN()
        net.load_state_dict(torch.load('./cptk_cnn/model_best.pth'))
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
    else:
        net = MindLSTMSep()
        net.load_state_dict(torch.load('./cptk_lstm_sep/model_best.pth'))
        seq_len = 5
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    # event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '../../regenerate_annotation/'
    annotation_path = '../../reformat_annotation/'
    color_img_path = '../../annotations/'
    feature_path = '../../data/feature_single/'
    # save_path = '/home/shuwen/data/data_preprocessing2/mind_baseline_output/'
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    # clips = os.listdir(event_label_path)
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
                mc = net(obj_patch, hog_input)
                max_score, idx_mc = torch.max(mc, 1)
                pre_combination = mind_combination[idx_mc.cpu().numpy()[0]]
                # max_score, idx_m21 = torch.max(m21, 1)
                # max_score, idx_m12 = torch.max(m12, 1)
                # max_score, idx_m1 = torch.max(m1, 1)
                # max_score, idx_m2 = torch.max(m2, 1)
                m1_predict.append(pre_combination[0])
                m2_predict.append(pre_combination[1])
                m12_predict.append(pre_combination[2])
                m21_predict.append(pre_combination[3])
                mc_predict.append(pre_combination[4])

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
                     [score1, score2, score3, score4, score5],
                     [cm1, cm2, cm12, cm21, cmc]], f)

if __name__ == '__main__':
    main()










