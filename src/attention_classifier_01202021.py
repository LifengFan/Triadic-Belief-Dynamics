import numpy as np
import torch
from torch.autograd import Variable
import pickle

from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import os
from sklearn import metrics
import glob
from sklearn import svm
import matplotlib.pyplot as plt
import joblib
# import seaborn as sn
import random
from torchvision import transforms
import cv2
import sys
# sys.path.append('/home/shuwen/projects/six_minds/data/Six-MInds-Project/data_processing_scripts/')
from metadata import *
from attention_model_01202021 import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors


def get_data(data_path):

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
    test_ratio = int(len(c)*0.2)
    data, label = zip(*c)

    train_x, train_y = data[:train_ratio], label[:train_ratio]
    validate_x, validate_y = data[train_ratio:train_ratio + test_ratio], label[train_ratio:train_ratio + test_ratio]
    test_x, test_y = data[train_ratio + test_ratio:], label[train_ratio + test_ratio:]


    return train_x, train_y, validate_x, validate_y, test_x, test_y

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
        img_input, hog_input = self.train_x[index]
        obj_patch = transforms.ToPILImage()(img_input)
        obj_patch = self.transforms(obj_patch).numpy()
        obj_patch_input = obj_patch

        actual_val = self.train_y[index]

        return obj_patch_input, hog_input, actual_val

    def __len__(self):
        return len(self.train_x)

def collate_fn(batch):
    N = len(batch)

    obj_batch = np.zeros((N, 3, 224, 224))
    hog_batch = np.zeros((N, 162*2))
    label_batch = np.zeros((N, 3))


    for i, (obj, hog, label) in enumerate(batch):
        obj_batch[i, ...] = obj
        hog_batch[i, ...] = hog
        label_batch[i,...] = label

    obj_batch = torch.FloatTensor(obj_batch)
    hog_batch = torch.FloatTensor(hog_batch)
    label_batch = torch.FloatTensor(label_batch)

    return obj_batch, hog_batch, label_batch

def main():

    data_path = './attention_training/'


    train_x, train_y, validate_x, validate_y, test_x, test_y = get_data(data_path)
    print(len(train_x), len(validate_x), len(test_x))

    learningRate = 0.01
    epochs = 300
    batch_size = 256

    # train(learningRate, epochs, batch_size, train_x, train_y, validate_x, validate_y, './cptk/74_47.pth', 20)
    net = AttMat()
    net.load_state_dict(torch.load('./cptk/model_best.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    test_score(net, test_x, test_y, batch_size, '', 'test')

def plot_confusion_matrix(cmc):
    df_cm = pd.DataFrame(cmc, range(cmc.shape[0]), range(cmc.shape[1]))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()


def test_score(net, data, label, batch_size, proj_name = None, dataset = None):

    net.eval()
    total_mc = np.empty(0)
    total_act_mc = np.empty(0)


    train_set = mydataset(data, label)
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size,
                                               shuffle=False)



    net.eval()

    pbar = tqdm(train_loader)
    total = 0
    correct = 0
    for batch in pbar:

        obj_batch, hog_batch, mc_label_batch = batch

        obj_batch = obj_batch.cuda()
        hog_batch = hog_batch.cuda()
        mc_label_batch = mc_label_batch.numpy()



        output = net(obj_batch, hog_batch)

        total_mc = output.data.cpu().numpy()
        total_mc = (total_mc > 0.5).astype(float)
        total_act_mc = mc_label_batch
        for data_id in range(total_mc.shape[0]):
            correct += np.sum(total_mc == total_act_mc)
            total += total_mc.shape[0]*total_mc.shape[1]

    if dataset:
        print(float(correct)/total)
    # if dataset:
    #     results_mc = metrics.classification_report(total_act_mc, total_mc, digits=3)
    #
    #     print(results_mc)
    #
    #
    #     cmc = metrics.confusion_matrix(total_act_mc, total_mc)
    #
    #
    #     plot_confusion_matrix(cmc)
    #
    #
    #     score1 = metrics.accuracy_score(total_act_mc, total_mc)
    #
    #     print([score1])
    #     with open('./cptk_' + proj_name + '/' + dataset + '.p', 'wb') as f:
    #         pickle.dump([results_mc], f)
    #
    # score1 = metrics.accuracy_score(total_act_mc, total_mc)
    return float(correct)/total

def train(learningRate, epochs, batch_size, train_x, train_y, validate_x, validate_y, checkpoint = None, startepoch = None):

    model = AttMat()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    if startepoch is not None:
        startepoch = startepoch
    else:
        startepoch = 0
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    losses = []
    best_score = 0


    train_set = mydataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size,
                                               shuffle=False)


    for epoch in range(startepoch, epochs):
        model.train()
        # training set -- perform model training
        epoch_training_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader)
        epoch_training_loss = 0
        for batch in pbar:
            obj_batch, hog_batch, label_batch = batch
            obj_batch = obj_batch.cuda()
            hog_batch = hog_batch.cuda()
            label_batch = label_batch.cuda()

            optimizer.zero_grad()

            output = model(obj_batch, hog_batch)
            loss = criterion(output, label_batch)
            loss.backward()

            optimizer.step()

            # calculating loss
            epoch_training_loss += loss.data.item()
            num_batches += 1

        # scheduler.step()
        print("epoch:{}/loss:{}".format(epoch, epoch_training_loss))
        losses.append(epoch_training_loss/num_batches)

        score = test_score(model, validate_x, validate_y, batch_size)
        if score > best_score:
            best_score = score
            print('best_score: {}'.format(score))
            save_path = './cptk/model_best.pth'
            torch.save(model.state_dict(), save_path)
        if epoch%50 == 0:
            for param in optimizer.param_groups:
                if param['lr'] > 1e-5:
                    param['lr'] = param['lr']*0.5

        if epoch%50 == 0:
            save_path = './cptk/model_' +str(epoch)+'.pth'
            torch.save(model.state_dict(), save_path)
            plt.plot(losses)
            plt.show()

def test_data_seq(clip):
    annotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    bbox_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    attmat_path = '../attention_classifier/person_attention/'
    save_path = './save_mat_3/' + clip.split('.')[0] + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    net = AttMat()
    net.load_state_dict(torch.load('./cptk/model_best.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])

    annt = pd.read_csv(bbox_path + clip.split('.')[0] + '.txt', sep=",", header=None)
    annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                    "label"]
    img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
    with open(feature_path + clip, 'rb') as f:
        features = pickle.load(f)
    with open('../mind_change_classifier/person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    with open(os.path.join(annotation_path, clip), 'rb') as f:
        obj_records = pickle.load(f)
    with open(os.path.join(attmat_path, clip), 'rb') as f:
        att_mat = pickle.load(f)

    if person_ids[clip.split('.')[0]] == 'P1':
        p1_hog = features[1]
        p2_hog = features[2]
    else:
        p1_hog = features[2]
        p2_hog = features[1]

    obj_names = sorted(obj_records.keys())
    obj_int_list = [int(obj_id[1:]) for obj_id in obj_names]
    obj_int_list = sorted(obj_int_list)
    # print(obj_int_list)
    obj_names = ['O{}'.format(obj_int) for obj_int in obj_int_list]
    count = 0
    for frame_id in range(len(obj_records[obj_names[0]])):
        att_matrix = np.zeros((2, len(obj_names) + 2))
        img = cv2.imread(img_names[frame_id])
        # for obj_id, obj_name in enumerate(obj_names):

        att_matrix[0, 1] = att_mat[frame_id]['p1']
        att_matrix[1, 0] = att_mat[frame_id]['p2']
        p1_obj = None
        p2_obj = None
        p1_score = 0
        p2_score = 0
        for obj_id in range(len(obj_names)):
            obj_name = obj_names[obj_id]
            obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
            assert obj_frame.shape[0] > 0
            x_min = obj_frame['x_min'].item()
            y_min = obj_frame['y_min'].item()
            x_max = obj_frame['x_max'].item()
            y_max = obj_frame['y_max'].item()

            img_patch = img[y_min:y_max, x_min:x_max]
            hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
            obj_patch = transforms.ToPILImage()(img_patch)
            obj_patch = obj_transforms(obj_patch).numpy()

            obj_batch = torch.FloatTensor(obj_patch)
            hog_batch = torch.FloatTensor(hog)

            obj_batch = obj_batch.cuda().view(1, 3, 224, 224)
            hog_batch = hog_batch.cuda().view(1, hog_batch.shape[0])

            output = net(obj_batch, hog_batch)

            output_ori = output.data.cpu().numpy()
            output = (output_ori > 0.8).astype(float).reshape(-1)
            # att_matrix[0, obj_id + 2] = output_ori[0,1]
            # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)
            #
            # att_matrix[1, obj_id + 2] = output_ori[0,2]
            # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)

            # if output[1] > 0:
            #     print(output_ori[0,1])
            #     att_matrix[0, obj_id] = output_ori[0,1]
            #     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)
            if output[2] > 0:
                att_matrix[1, obj_id] = output_ori[0,2]
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
            # if output[0] > 0:
            #     att_matrix[1, obj_id] = output_ori[0, 0]
            #     att_matrix[0, obj_id] = output_ori[0, 0]
            #     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
            # if att_matrix[0, 1] > 0:
            #     obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == 'P2')]
            #     assert obj_frame.shape[0] > 0
            #     x_min = obj_frame['x_min'].item()
            #     y_min = obj_frame['y_min'].item()
            #     x_max = obj_frame['x_max'].item()
            #     y_max = obj_frame['y_max'].item()
            #     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)
            if att_matrix[1, 0] > 0:
                obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == 'P1')]
                assert obj_frame.shape[0] > 0
                x_min = obj_frame['x_min'].item()
                y_min = obj_frame['y_min'].item()
                x_max = obj_frame['x_max'].item()
                y_max = obj_frame['y_max'].item()
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)

        cv2.imshow('img',  img)
        cv2.waitKey(20)
        # start = 0xF0D20F  # From
        # stop = 0x00994C  # To
        # num = 100  # Divided into 100 steps
        # color = ["#{:02x}{:02x}{:02x}".format(x, y, z) for x, y, z in zip(
        #     np.round(np.linspace(start >> 16, stop >> 16, num)).astype(int),
        #     np.round(np.linspace((start >> 8) & 0xFF, (stop >> 8) & 0xFF, num)).astype(int),
        #     np.round(np.linspace(start & 0xFF, stop & 0xFF, num)).astype(int))]
        # sns.set(font_scale=2)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # y_labels = ['P1', 'P2'] + ['O{}'.format(obj_id + 1) for obj_id in range(len(obj_names))]
        # # print(len(obj_names))
        # x_labels = ['P1', 'P2']
        # cax = ax.matshow(att_matrix.transpose(), vmin = 0, vmax = 1)
        # # cax = sns.heatmap(att_matrix.transpose(), xticklabels=x_labels, yticklabels=y_labels, square=True, vmin=0, vmax=1,
        # #                   cmap = color)
        #
        # # plt.yticks(rotation = 0)
        # # plt.xticks(rotation=0)
        # fig.colorbar(cax)
        #
        # ax_dict = {'fontsize': 18}
        # ax.set_xticks(np.arange(len(x_labels)))
        # ax.set_yticks(np.arange(len(y_labels)))
        # ax.set_xticklabels(x_labels, fontdict=ax_dict)
        # ax.set_yticklabels(y_labels, fontdict=ax_dict)
        # # ax.axis('image')
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(img)
        # plt.savefig(save_path + img_names[frame_id].split('/')[-1])
        # plt.close()

def test_data_seq_get_obj_per_frame(clip, net):
    print(clip)

    annotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    bbox_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    attmat_path = '../attention_classifier/person_attention/'
    save_path = './attention_obj_name/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.exists(save_path + clip):
        return

    if not os.path.exists(bbox_path + clip.split('.')[0] + '.txt'):
        return

    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])

    annt = pd.read_csv(bbox_path + clip.split('.')[0] + '.txt', sep=",", header=None)
    annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                    "label"]
    img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
    with open(feature_path + clip, 'rb') as f:
        features = pickle.load(f)
    with open('../mind_change_classifier/person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    with open(os.path.join(annotation_path, clip), 'rb') as f:
        obj_records = pickle.load(f)
    # with open(os.path.join(attmat_path, clip), 'rb') as f:
    #     att_mat = pickle.load(f)

    if person_ids[clip.split('.')[0]] == 'P1':
        p1_hog = features[1]
        p2_hog = features[2]
    else:
        p1_hog = features[2]
        p2_hog = features[1]

    obj_names = sorted(obj_records.keys())
    obj_int_list = [int(obj_id[1:]) for obj_id in obj_names]
    obj_int_list = sorted(obj_int_list)
    # print(obj_int_list)
    obj_names = ['O{}'.format(obj_int) for obj_int in obj_int_list]
    count = 0
    att_names = {'P1':[], 'P2':[]}
    for frame_id in range(len(obj_records[obj_names[0]])):
        att_matrix = np.zeros((2, len(obj_names) + 2))
        img = cv2.imread(img_names[frame_id])
        # for obj_id, obj_name in enumerate(obj_names):

        # att_matrix[0, 1] = att_mat[frame_id]['p1']
        # att_matrix[1, 0] = att_mat[frame_id]['p2']
        p1_obj = None
        p2_obj = None
        p1_score = 0
        p2_score = 0
        for obj_id in range(len(obj_names)):
            obj_name = obj_names[obj_id]
            obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
            assert obj_frame.shape[0] > 0
            x_min = obj_frame['x_min'].item()
            y_min = obj_frame['y_min'].item()
            x_max = obj_frame['x_max'].item()
            y_max = obj_frame['y_max'].item()

            img_patch = img[y_min:y_max, x_min:x_max]
            hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
            obj_patch = transforms.ToPILImage()(img_patch)
            obj_patch = obj_transforms(obj_patch).numpy()

            obj_batch = torch.FloatTensor(obj_patch)
            hog_batch = torch.FloatTensor(hog)

            obj_batch = obj_batch.cuda().view(1, 3, 224, 224)
            hog_batch = hog_batch.cuda().view(1, hog_batch.shape[0])

            output = net(obj_batch, hog_batch)

            output_ori = output.data.cpu().numpy()
            if output_ori[0, 1] > p1_score:
                p1_obj = obj_name
                p1_score = output_ori[0, 1]
            if output_ori[0, 2] > p2_score:
                p2_obj = obj_name
                p2_score = output_ori[0, 2]

        att_names['P1'].append(p1_obj)
        att_names['P2'].append(p2_obj)
    with open(save_path + clip, 'wb') as f:
        pickle.dump(att_names, f)
        # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == p1_obj)]
        # assert obj_frame.shape[0] > 0
        # x_min = obj_frame['x_min'].item()
        # y_min = obj_frame['y_min'].item()
        # x_max = obj_frame['x_max'].item()
        # y_max = obj_frame['y_max'].item()
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
        #
        # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == p2_obj)]
        # assert obj_frame.shape[0] > 0
        # x_min = obj_frame['x_min'].item()
        # y_min = obj_frame['y_min'].item()
        # x_max = obj_frame['x_max'].item()
        # y_max = obj_frame['y_max'].item()
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)
        #
        # cv2.imshow('img',  img)
        # cv2.waitKey(20)
        # start = 0xF0D20F  # From
        # stop = 0x00994C  # To
        # num = 100  # Divided into 100 steps
        # color = ["#{:02x}{:02x}{:02x}".format(x, y, z) for x, y, z in zip(
        #     np.round(np.linspace(start >> 16, stop >> 16, num)).astype(int),
        #     np.round(np.linspace((start >> 8) & 0xFF, (stop >> 8) & 0xFF, num)).astype(int),
        #     np.round(np.linspace(start & 0xFF, stop & 0xFF, num)).astype(int))]
        # sns.set(font_scale=2)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # y_labels = ['P1', 'P2'] + ['O{}'.format(obj_id + 1) for obj_id in range(len(obj_names))]
        # # print(len(obj_names))
        # x_labels = ['P1', 'P2']
        # cax = ax.matshow(att_matrix.transpose(), vmin = 0, vmax = 1)
        # # cax = sns.heatmap(att_matrix.transpose(), xticklabels=x_labels, yticklabels=y_labels, square=True, vmin=0, vmax=1,
        # #                   cmap = color)
        #
        # # plt.yticks(rotation = 0)
        # # plt.xticks(rotation=0)
        # fig.colorbar(cax)
        #
        # ax_dict = {'fontsize': 18}
        # ax.set_xticks(np.arange(len(x_labels)))
        # ax.set_yticks(np.arange(len(y_labels)))
        # ax.set_xticklabels(x_labels, fontdict=ax_dict)
        # ax.set_yticklabels(y_labels, fontdict=ax_dict)
        # # ax.axis('image')
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(img)
        # plt.savefig(save_path + img_names[frame_id].split('/')[-1])
        # plt.close()

def test_data_seq_get_att_vec_per_frame(clip, net):
    print(clip)

    annotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    bbox_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    attmat_path = '../attention_classifier/person_attention/'
    save_path = './att_vec/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.exists(save_path + clip):
        return

    if not os.path.exists(bbox_path + clip.split('.')[0] + '.txt'):
        return

    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])

    annt = pd.read_csv(bbox_path + clip.split('.')[0] + '.txt', sep=",", header=None)
    annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                    "label"]
    img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
    with open(feature_path + clip, 'rb') as f:
        features = pickle.load(f)
    with open('../mind_change_classifier/person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    with open(os.path.join(annotation_path, clip), 'rb') as f:
        obj_records = pickle.load(f)
    # with open(os.path.join(attmat_path, clip), 'rb') as f:
    #     att_mat = pickle.load(f)

    if person_ids[clip.split('.')[0]] == 'P1':
        p1_hog = features[1]
        p2_hog = features[2]
    else:
        p1_hog = features[2]
        p2_hog = features[1]

    obj_names = sorted(obj_records.keys())
    obj_int_list = [int(obj_id[1:]) for obj_id in obj_names]
    obj_int_list = sorted(obj_int_list)
    # print(obj_int_list)
    obj_names = ['O{}'.format(obj_int) for obj_int in obj_int_list]
    count = 0
    att_vec = {obj_name:[] for obj_name in obj_names}
    for frame_id in range(len(obj_records[obj_names[0]])):
        att_matrix = np.zeros((2, len(obj_names) + 2))
        img = cv2.imread(img_names[frame_id])

        for obj_id in range(len(obj_names)):
            obj_name = obj_names[obj_id]
            obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
            assert obj_frame.shape[0] > 0
            x_min = obj_frame['x_min'].item()
            y_min = obj_frame['y_min'].item()
            x_max = obj_frame['x_max'].item()
            y_max = obj_frame['y_max'].item()

            img_patch = img[y_min:y_max, x_min:x_max]
            hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
            obj_patch = transforms.ToPILImage()(img_patch)
            obj_patch = obj_transforms(obj_patch).numpy()

            obj_batch = torch.FloatTensor(obj_patch)
            hog_batch = torch.FloatTensor(hog)

            obj_batch = obj_batch.cuda().view(1, 3, 224, 224)
            hog_batch = hog_batch.cuda().view(1, hog_batch.shape[0])

            output = net(obj_batch, hog_batch)

            output_ori = output.data.cpu().numpy()
            att_vec[obj_name].append(output_ori)


    with open(save_path + clip, 'wb') as f:
        pickle.dump(att_vec, f)

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

def test_data_seq_get_att_vec_per_frame_raw_obj(clip, net):
    print(clip)
    annotation_path = '/home/lfan/Dropbox/Projects/NIPS20/regenerate_annotation/'
    bbox_path = '/home/lfan/Dropbox/Projects/NIPS20/reformat_annotation/'
    feature_path = '/home/lfan/Dropbox/Projects/NIPS20/data/feature_single/'
    color_img_path = '/home/lfan/Dropbox/Projects/NIPS20/annotations/'
    # attmat_path = '../attention_classifier/person_attention/'
    obj_bbox_path = '/home/lfan/Dropbox/Projects/NIPS20/data/interpolate_bbox/'
    save_path = '/home/lfan/Dropbox/Projects/NIPS20/code/att_vec_raw_obj_01202021/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.exists(save_path + clip):
        return

    if not os.path.exists(bbox_path + clip.split('.')[0] + '.txt'):
        return

    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])

    annt = pd.read_csv(bbox_path + clip.split('.')[0] + '.txt', sep=",", header=None)
    annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                    "label"]
    img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
    with open(feature_path + clip, 'rb') as f:
        features = pickle.load(f)
    with open('/home/lfan/Dropbox/Projects/NIPS20/data/person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    with open(os.path.join(annotation_path, clip), 'rb') as f:
        obj_records = pickle.load(f)
    # with open(os.path.join(attmat_path, clip), 'rb') as f:
    #     att_mat = pickle.load(f)

    if person_ids[clip.split('.')[0]] == 'P1':
        p1_hog = features[1]
        p2_hog = features[2]
    else:
        p1_hog = features[2]
        p2_hog = features[1]

    obj_names = sorted(obj_records.keys())
    obj_int_list = [int(obj_id[1:]) for obj_id in obj_names]
    obj_int_list = sorted(obj_int_list)
    # print(obj_int_list)
    obj_names = glob.glob(obj_bbox_path + clip.split('.')[0] + '/*.p')
    count = 0
    att_vec = {obj_name:[] for obj_name in obj_names}
    for frame_id in range(len(img_names)):
        att_matrix = np.zeros((2, len(obj_names) + 2))
        img = cv2.imread(img_names[frame_id])

        for obj_id in range(len(obj_names)):
            obj_name = obj_names[obj_id]
            with open(obj_name, 'rb') as f:
                obj_bboxs = pickle.load(f)
            obj_bbox = obj_bboxs[frame_id]
            x_min = int(obj_bbox[0])
            y_min = int(obj_bbox[1])
            x_max = int(obj_bbox[0] + obj_bbox[2])
            y_max = int(obj_bbox[1] + obj_bbox[3])
            # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
            # assert obj_frame.shape[0] > 0
            # x_min = obj_frame['x_min'].item()
            # y_min = obj_frame['y_min'].item()
            # x_max = obj_frame['x_max'].item()
            # y_max = obj_frame['y_max'].item()
            # cv2.rectangle(img, (int(obj_bbox[0]), int(obj_bbox[1])),
            #               (int(obj_bbox[2] + obj_bbox[0]), int(obj_bbox[3] + obj_bbox[1])), (255, 0, 0), thickness=2)
            # cv2.imshow('img', img)
            # cv2.waitKey(200)

            y_max = max(1, y_max)
            y_max = min(720, y_max)
            y_min = max(0, y_min)
            y_min = min(719, y_min)

            x_min = max(0, x_min)
            x_min = min(1279, x_min)
            x_max = max(1, x_max)
            x_max = min(1280, x_max)

            img_patch = img[y_min:y_max, x_min:x_max]

            hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
            obj_patch = transforms.ToPILImage()(img_patch)
            obj_patch = obj_transforms(obj_patch).numpy()

            obj_batch = torch.FloatTensor(obj_patch)
            hog_batch = torch.FloatTensor(hog)

            obj_batch = obj_batch.cuda().view(1, 3, 224, 224)
            hog_batch = hog_batch.cuda().view(1, hog_batch.shape[0])

            output = net(obj_batch, hog_batch)

            output_ori = output.data.cpu().numpy()
            att_vec[obj_name].append(output_ori)


    with open(save_path + clip, 'wb') as f:
        pickle.dump(att_vec, f)

def test_data_seq_get_obj_per_frame_raw_obj(clip, net):
    print(clip)
    annotation_path = '/home/lfan/Dropbox/Projects/NIPS20/regenerate_annotation/'
    bbox_path = '/home/lfan/Dropbox/Projects/NIPS20/reformat_annotation/'
    feature_path = '/home/lfan/Dropbox/Projects/NIPS20/data/feature_single/'
    color_img_path = '/home/lfan/Dropbox/Projects/NIPS20/annotations/'
    # attmat_path = '../attention_classifier/person_attention/'
    obj_bbox_path = '/home/lfan/Dropbox/Projects/NIPS20/data/interpolate_bbox/'
    save_path = '/home/lfan/Dropbox/Projects/NIPS20/code/att_obj_id_w_raw_objs_01202021/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(save_path + clip):
        return
    if not os.path.exists(bbox_path + clip.split('.')[0] + '.txt'):
        return
    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    annt = pd.read_csv(bbox_path + clip.split('.')[0] + '.txt', sep=",", header=None)
    annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                    "label"]
    img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
    with open(feature_path + clip, 'rb') as f:
        features = pickle.load(f)
    with open('/home/lfan/Dropbox/Projects/NIPS20/data/person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    with open(os.path.join(annotation_path, clip), 'rb') as f:
        obj_records = pickle.load(f)
    # with open(os.path.join(attmat_path, clip), 'rb') as f:
    #     att_mat = pickle.load(f)
    if person_ids[clip.split('.')[0]] == 'P1':
        p1_hog = features[1]
        p2_hog = features[2]
    else:
        p1_hog = features[2]
        p2_hog = features[1]
    obj_names = sorted(glob.glob(obj_bbox_path + clip.split('.')[0] + '/*.p'))
    count = 0
    att_names = {'P1':[], 'P2':[]}
    for frame_id in range(clips_len[clip]):
        att_matrix = np.zeros((2, len(obj_names) + 2))
        img = cv2.imread(img_names[frame_id])
        # for obj_id, obj_name in enumerate(obj_names):
        # att_matrix[0, 1] = att_mat[frame_id]['p1']
        # att_matrix[1, 0] = att_mat[frame_id]['p2']
        p1_obj = None
        p2_obj = None
        p1_score = 0
        p2_score = 0
        for obj_id in range(len(obj_names)):
            obj_name = obj_names[obj_id]
            with open(obj_name, 'rb') as f:
                obj_bboxs = pickle.load(f)
            obj_bbox = obj_bboxs[frame_id]
            x_min = int(obj_bbox[0])
            y_min = int(obj_bbox[1])
            x_max = int(obj_bbox[0] + obj_bbox[2])
            y_max = int(obj_bbox[1] + obj_bbox[3])
            y_max = max(1, y_max)
            y_max = min(720, y_max)
            y_min = max(0, y_min)
            y_min = min(719, y_min)

            x_min = max(0, x_min)
            x_min = min(1279, x_min)
            x_max = max(1, x_max)
            x_max = min(1280, x_max)

            img_patch = img[y_min:y_max, x_min:x_max]
            hog = np.hstack([p1_hog[frame_id][-162 - 10:-10], p2_hog[frame_id][-162 - 10:-10]])
            obj_patch = transforms.ToPILImage()(img_patch)
            obj_patch = obj_transforms(obj_patch).numpy()
            obj_batch = torch.FloatTensor(obj_patch)
            hog_batch = torch.FloatTensor(hog)
            obj_batch = obj_batch.cuda().view(1, 3, 224, 224)
            hog_batch = hog_batch.cuda().view(1, hog_batch.shape[0])
            output = net(obj_batch, hog_batch)
            output_ori = output.data.cpu().numpy()
            if output_ori[0, 1] > p1_score:
                p1_obj = obj_name
                p1_score = output_ori[0, 1]
            if output_ori[0, 2] > p2_score:
                p2_obj = obj_name
                p2_score = output_ori[0, 2]
        att_names['P1'].append(p1_obj)
        att_names['P2'].append(p2_obj)
    with open(save_path + clip, 'wb') as f:
        pickle.dump(att_names, f)
        # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == p1_obj)]
        # assert obj_frame.shape[0] > 0
        # x_min = obj_frame['x_min'].item()
        # y_min = obj_frame['y_min'].item()
        # x_max = obj_frame['x_max'].item()
        # y_max = obj_frame['y_max'].item()
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
        #
        # obj_frame = annt.loc[(annt.frame == frame_id) & (annt.name == p2_obj)]
        # assert obj_frame.shape[0] > 0
        # x_min = obj_frame['x_min'].item()
        # y_min = obj_frame['y_min'].item()
        # x_max = obj_frame['x_max'].item()
        # y_max = obj_frame['y_max'].item()
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)
        #
        # cv2.imshow('img',  img)
        # cv2.waitKey(20)
        # start = 0xF0D20F  # From
        # stop = 0x00994C  # To
        # num = 100  # Divided into 100 steps
        # color = ["#{:02x}{:02x}{:02x}".format(x, y, z) for x, y, z in zip(
        #     np.round(np.linspace(start >> 16, stop >> 16, num)).astype(int),
        #     np.round(np.linspace((start >> 8) & 0xFF, (stop >> 8) & 0xFF, num)).astype(int),
        #     np.round(np.linspace(start & 0xFF, stop & 0xFF, num)).astype(int))]
        # sns.set(font_scale=2)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # y_labels = ['P1', 'P2'] + ['O{}'.format(obj_id + 1) for obj_id in range(len(obj_names))]
        # # print(len(obj_names))
        # x_labels = ['P1', 'P2']
        # cax = ax.matshow(att_matrix.transpose(), vmin = 0, vmax = 1)
        # # cax = sns.heatmap(att_matrix.transpose(), xticklabels=x_labels, yticklabels=y_labels, square=True, vmin=0, vmax=1,
        # #                   cmap = color)
        #
        # # plt.yticks(rotation = 0)
        # # plt.xticks(rotation=0)
        # fig.colorbar(cax)
        #
        # ax_dict = {'fontsize': 18}
        # ax.set_xticks(np.arange(len(x_labels)))
        # ax.set_yticks(np.arange(len(y_labels)))
        # ax.set_xticklabels(x_labels, fontdict=ax_dict)
        # ax.set_yticklabels(y_labels, fontdict=ax_dict)
        # # ax.axis('image')
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(img)
        # plt.savefig(save_path + img_names[frame_id].split('/')[-1])
        # plt.close()

if __name__ == '__main__':
    # main()
    net = AttMat()
    net.load_state_dict(torch.load('./model_best_01202021.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for clip in mind_test_clips:
        test_data_seq_get_att_vec_per_frame_raw_obj(clip, net)
        # test_data_seq_get_obj_per_frame_raw_obj(clip,net)










