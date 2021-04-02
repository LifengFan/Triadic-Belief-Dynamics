from test_communication_event import *
import pickle
import joblib
import sys
from metadata import *
import numpy as np
from torchvision import transforms
import glob
import cv2
import random

def find_data_seq_with_label(attmat_obj, start_id, video_len, category, clip, tracker_bbox, battery_bbox, tracker_event_labels):
    slide_size = 1
    test_seq = []
    event_labels = []
    attmat, obj_list = attmat_obj
    corred = {0:0, 1:4, 2:5}
    while (start_id < video_len):
        event_label = tracker_event_labels[start_id:start_id + 5]
        min_label = np.min(event_label)
        if np.sum(event_label - min_label) > 0:
            start_id += slide_size
            continue
        if tracker_event_labels[start_id] == 3:
            start_id += slide_size
            continue
        obj_list_seq = check_atom(attmat, start_id, video_len, obj_list, category, clip, tracker_bbox, battery_bbox)
        if obj_list_seq:
            # print([clip, start_id, obj_list_seq, obj_list])
            # if tracker_event_labels[start_id]>0:
            #     test_seq.append([clip, start_id, obj_list_seq, obj_list])
            #     event_labels.append(tracker_event_labels[start_id])

            test_seq.append([clip, start_id, obj_list_seq, obj_list])
            event_labels.append(tracker_event_labels[start_id])
            print(tracker_event_labels[start_id:start_id + 5])
            start_id = min(start_id + 5, video_len + 1)
        else:
            start_id += slide_size
    return test_seq, event_labels

def seg2frame(segs):
    event_labels = np.zeros(segs[-1][1])
    for seg in segs:
        event_labels[seg[0]:seg[1] + 1] = seg[2]
    return event_labels

class mydataset_atomic_with_label(torch.utils.data.Dataset):
    def __init__(self, seq, label):
        self.seq = seq
        self.label = label
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.img_path = '../annotations/'
        self.person_tracker_bbox = '../data/3d_pose2gaze/tracker_record_bbox/'
        self.person_battery_bbox = '../data/3d_pose2gaze/record_bbox/'
        self.obj_bbox = '../post_neighbor_smooth_newseq/'
        self.cate_path = '../data/track_cate/'

    def __getitem__(self, index):
        rec = self.seq[index]
        label = self.label[index]
        head_patch_sq = np.zeros((self.seq_size, 4, 3, 224, 224)) # [5,4,3,224,224]
        pos_sq = np.zeros((self.seq_size, self.node_num, 6)) #[5,4,6]
        attmat_sq = np.zeros((self.seq_size, self.node_num, self.node_num)) #[5,4,4]
        clip, start_id, obj_list_seq, obj_list = rec
        img_names = sorted(glob.glob(os.path.join(self.img_path, clip.split('.')[0], 'kinect/*.jpg')))
        with open(self.person_tracker_bbox + clip, 'rb') as f:
            tracker_bbox = joblib.load(f)
        with open(self.person_battery_bbox + clip, 'rb') as f:
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

        return head_patch_sq, pos_sq, attmat_sq, index, label

    def __len__(self):
        return len(self.seq)

class mydataset_atomic_with_label_first_view(torch.utils.data.Dataset):
    def __init__(self, seq, label, args):
        self.seq = seq
        self.label = label
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
        label = self.label[index]
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

        return head_patch_sq, pos_sq, attmat_sq, index, hog_sq, label

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

    for i, (head_patch_sq, pos_sq, attmat_sq, index_sq, hog_sq, label) in enumerate(batch):
            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq
            attmat_batch[i, ...] = attmat_sq
            index_batch[i,...]=index_sq
            label_batch[i, ...] = label
            hog_batch[i, ...] = hog_sq

    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)
    index_batch = torch.FloatTensor(index_batch)
    hog_batch = torch.FloatTensor(hog_batch)
    label_batch = torch.LongTensor(label_batch)

    return head_batch, pos_batch, attmat_batch, index_batch, hog_batch, label_batch

def collate_fn_atomic_fine_tune(batch):
    N = len(batch)
    max_node_num = 4
    sq_len=5

    head_batch = np.zeros((N, sq_len, 4, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))
    attmat_batch = np.zeros((N, sq_len, max_node_num, max_node_num))
    index_batch = np.zeros((N,  sq_len))
    label_batch = np.zeros(N)

    for i, (head_patch_sq, pos_sq, attmat_sq, index_sq, label) in enumerate(batch):
            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq
            attmat_batch[i, ...] = attmat_sq
            index_batch[i,...]=index_sq
            label_batch[i, ...] = label

    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)
    index_batch = torch.FloatTensor(index_batch)
    label_batch = torch.LongTensor(label_batch)

    return head_batch, pos_batch, attmat_batch, index_batch, label_batch

def get_data(args):

    seg_path = args.seg_label # '../data/segment_labels/'
    attmat_path = args.attmat_path #'../data/record_attention_matrix/'
    cate_path = args.cate_path # '../data/track_cate/'
    person_tracker_bbox = args.tracker_bbox # '../data/3d_pose2gaze/tracker_record_bbox/'
    person_battery_bbox = args.battery_bbox #'../data/3d_pose2gaze/record_bbox/'

    clips = os.listdir(seg_path)
    model = Atomic_node_only_lstm()
    args.cuda = True
    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
    model = load_best_checkpoint(model, path='.')

    test_seqs = []
    event_labels = []
    for clip in clips:
        print(clip)
        if not clip.split('.')[0] in event_seg_tracker:
            continue

        with open(seg_path + clip, 'rb') as f:
            seg_dicts = pickle.load(f, encoding='latin1')
        with open(attmat_path + clip, 'rb') as f:
            attmat_obj = pickle.load(f, encoding='latin1')
        with open(cate_path + clip, 'rb') as f:
            category = pickle.load(f, encoding='latin1')
        with open(person_tracker_bbox + clip, 'rb') as f:
            tracker_bbox = pickle.load(f, encoding='latin1')
        with open(person_battery_bbox + clip, 'rb') as f:
            battery_bbox = pickle.load(f, encoding='latin1')

        tracker_event_label_frame = seg2frame(event_seg_tracker[clip.split('.')[0]])
        battery_event_label_frame = seg2frame(event_seg_battery[clip.split(".")[0]])
        print(tracker_event_label_frame.shape[0], battery_event_label_frame.shape[0])
        assert tracker_event_label_frame.shape[0] == battery_event_label_frame.shape[0]

        test_seq, event_label = find_data_seq_with_label(attmat_obj, 0, len(tracker_bbox),
                                      category, clip, tracker_bbox, battery_bbox, tracker_event_label_frame)
        if len(test_seq) == 0:
            continue

        test_seqs.extend(test_seq)
        event_labels.extend(event_label)

    assert len(test_seqs) == len(event_labels)

    cal_dis = {}
    for label_i in event_labels:
        if label_i in cal_dis:
            cal_dis[label_i] += 1
        else:
            cal_dis[label_i] = 1
    print(cal_dis)

    c = list(zip(test_seqs, event_labels))
    random.shuffle(c)
    test_seqs, event_labels = zip(*c)
    train_ratio = int(len(test_seqs)*0.6)
    validate_ratio = int(len(test_seqs)*0.2)
    train_set = mydataset_atomic_with_label_first_view(test_seqs[:train_ratio], event_labels[:train_ratio], args)
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn_atomic_first_view, batch_size=16,shuffle=False)
    validate_set = mydataset_atomic_with_label_first_view(test_seqs[train_ratio:train_ratio + validate_ratio], event_labels[train_ratio:train_ratio + validate_ratio], args)
    validate_loader = torch.utils.data.DataLoader(validate_set, collate_fn=collate_fn_atomic_first_view, batch_size=16, shuffle=False)
    test_set = mydataset_atomic_with_label_first_view(test_seqs[train_ratio + validate_ratio:],event_labels[train_ratio + validate_ratio:],args)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic_first_view, batch_size=16,shuffle=False)

    with open('fine_tune_input_3.p', 'wb') as f:
        pickle.dump([train_loader, validate_loader, test_loader], f)


def parse_arguments():
    project_name = 'test atomic event'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--project-name', default=project_name, help='project name')
    home_path='/home/lfan/Dropbox/Projects/NIPS20/'
    home_path2='/media/lfan/HDD/NIPS20/'
    parser.add_argument('--project-path',default = home_path)
    parser.add_argument('--project-path2', default=home_path2)
    parser.add_argument('--data-path', default=home_path+'data/')
    parser.add_argument('--data-path2', default=home_path2 + 'data/')
    parser.add_argument('--img-path', default=home_path+'annotations/')
    parser.add_argument('--save-root', default='/media/lfan/HDD/NIPS20/Result/BeamSearch/')
    parser.add_argument('--save-path', default='/media/lfan/HDD/NIPS20/Result/BeamSearch/')
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

    # model settings
    parser.add_argument('--small-attr-class-num', type=int, default=7, help='small attribute class number')
    parser.add_argument('--big-attr-class-num', type=int, default=6, help='big attribute class number')
    parser.add_argument('--roi-feature-size', type=int, default=64 * 3, help='node and edge feature size')
    parser.add_argument('--message-size', type=int, default=12, help='message size of the message function')
    parser.add_argument('--lstm-seq-size', type=int, default=15, help='lstm sequence length')
    parser.add_argument('--lstm-hidden-size', type=int, default=500, help='hiddden state size of lstm')
    parser.add_argument('--link-hidden-size', type=int, default=1024, help='link hidden size of the link function')
    parser.add_argument('--link-hidden-layers', type=int, default=2, help='link hidden layers of the link function')
    parser.add_argument('--propagate-layers', type=int, default=2, help='propagate layers for message passing')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    get_data(args)