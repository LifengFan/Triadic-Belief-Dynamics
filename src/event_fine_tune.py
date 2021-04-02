import os
from Atomic_node_only_lstm import Atomic_node_only_lstm
import torch
import utils
import pickle
import joblib
import sys
sys.path.append('/home/shuwen/data/Six-Minds-Project/data_processing_scripts')
from metadata import *
from overall_event_get_input import *
import argparse
import pandas as pd

class mydataset_atomic_first_view_new_att(torch.utils.data.Dataset):
    def __init__(self, seq):
        self.seq = seq

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
        self.bbox_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
        self.cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
        self.feature_path = '/home/shuwen/data/data_preprocessing2/feature_single/'
        with open('../mind_change_classifier/person_id.p', 'rb') as f:
            self.person_ids = pickle.load(f)
        self.tracker_gt_path = '/home/shuwen/data/data_preprocessing2/tracker_gt_smooth/'


    def __getitem__(self, index):
        rec = self.seq[index]
        head_patch_sq = np.zeros((self.seq_size, 6, 3, 224, 224)) # [5,4,3,224,224]
        pos_sq = np.zeros((self.seq_size, self.node_num, 6)) #[5,4,6]
        clip, start_id, obj_list_seq = rec
        img_names = sorted(glob.glob(os.path.join(self.img_path, clip.split('.')[0], 'kinect/*.jpg')))
        tracker_img_names = sorted(glob.glob(os.path.join(self.img_path, clip.split('.')[0], 'tracker/*.jpg')))
        battery_img_names = sorted(glob.glob(os.path.join(self.img_path, clip.split('.')[0], 'battery/*.jpg')))
        annt = pd.read_csv(self.bbox_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        with open(os.path.join(self.tracker_gt_path, clip + '.p')) as f:
            gazes = pickle.load(f)


        for sq_id in range(len(obj_list_seq['P1'])):
            fid = start_id + sq_id
            img_name = img_names[fid]
            img = cv2.imread(img_name)
            tracker_img = cv2.imread(tracker_img_names[fid])
            battery_img  =cv2.imread(battery_img_names[fid])
            for node_i in [0, 1]:
                if node_i == 0:
                    obj_frame = annt.loc[(annt.frame == fid) & (annt.name == 'P1')]

                    x_min = obj_frame['x_min'].item()
                    y_min = obj_frame['y_min'].item()
                    x_max = obj_frame['x_max'].item()
                    y_max = obj_frame['y_max'].item()
                    head_box = [x_min, y_min, x_max, y_max]
                else:
                    obj_frame = annt.loc[(annt.frame == fid) & (annt.name == 'P2')]

                    x_min = obj_frame['x_min'].item()
                    y_min = obj_frame['y_min'].item()
                    x_max = obj_frame['x_max'].item()
                    y_max = obj_frame['y_max'].item()
                    head_box = [x_min, y_min, x_max, y_max]

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


                box_height = img.shape[0] / 6
                box_width = img.shape[1] / 6
                gaze_center = gazes[fid]
                top = gaze_center[1] - box_height
                left = gaze_center[0] - box_width
                top = max(0, top)
                left = max(0, left)
                top = min(img.shape[0] - box_height, top)
                left = min(img.shape[1] - box_width, left)
                bottom = gaze_center[1] + box_height
                right = gaze_center[0] + box_width
                bottom = min(img.shape[0], bottom)
                right = min(img.shape[1], right)
                bottom = max(box_height, bottom)
                right = max(box_width, right)

                # print(top, bottom, left, right)
                img_patch = tracker_img[int(top):int(bottom), int(left):int(right)]
                img_patch = cv2.resize(img_patch, (224, 224))  # .reshape((3, 224, 224))
                head_tracker = self.transforms(img_patch).numpy()
                for c in [0, 1, 2]:
                    head_tracker[c, :, :] = (head_tracker[c, :, :] - 0.5) / 0.5


                top = battery_img.shape[0] / 3 * 2
                left = battery_img.shape[1] / 3
                bottom = battery_img.shape[0]
                right = battery_img.shape[1] / 3 * 2
                img_patch = battery_img[top:bottom, left:right]
                img_patch = cv2.resize(img_patch, (224, 224))  # .reshape((3, 224, 224))
                head_battery = self.transforms(img_patch).numpy()
                for c in [0, 1, 2]:
                    head_battery[c, :, :] = (head_battery[c, :, :] - 0.5) / 0.5

                if self.person_ids[clip.split('.')[0]] == 'P1':
                    head_patch_sq[sq_id, 4, ...] = head_tracker
                    head_patch_sq[sq_id, 5, ...] = head_battery
                if self.person_ids[clip.split('.')[0]] == 'P2':
                    head_patch_sq[sq_id, 4, ...] = head_battery
                    head_patch_sq[sq_id, 5, ...] = head_tracker

            for pid, pname in enumerate(['P1', 'P2']):
                obj_name = obj_list_seq[pname][sq_id]
                obj_frame = annt.loc[(annt.frame == fid) & (annt.name == obj_name)]

                x_min = obj_frame['x_min'].item()
                y_min = obj_frame['y_min'].item()
                x_max = obj_frame['x_max'].item()
                y_max = obj_frame['y_max'].item()
                head_box = [x_min, y_min, x_max, y_max]
                img_patch = img[int(head_box[1]):int(head_box[3]), int(head_box[0]): int(head_box[2])]
                # cv2.rectangle(img, (int(head_box[0]), int(head_box[1])),
                #                                         (int(head_box[2]), int(head_box[3])),
                #                                         (255, 0, 0), thickness=3)
                img_patch = cv2.resize(img_patch, (224, 224))#.reshape((3, 224, 224))
                head = self.transforms(img_patch).numpy()
                for c in [0, 1, 2]:
                    head[c, :, :] = (head[c, :, :] - 0.5) / 0.5
                head_patch_sq[sq_id, 2, ...] = head
                pos_vec = np.array([head_box[0] / img.shape[1], head_box[1] / img.shape[0], (head_box[2]) / img.shape[1],
                                    (head_box[3]) / img.shape[0], (head_box[0] + head_box[2]) / 2 / img.shape[1],
                                    (head_box[1] + head_box[3]) / 2 / img.shape[0]])

                pos_sq[sq_id, 2 + pid, :] = pos_vec
            # cv2.imshow('img', img)
            # cv2.waitKey(20)
        return head_patch_sq, pos_sq

    def __len__(self):
        return len(self.seq)

def collate_fn_atomic_first_view_new_att(batch):
    N = len(batch)
    max_node_num = 4
    sq_len=5

    head_batch = np.zeros((N, sq_len, 6, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))

    for i, (head_patch_sq, pos_sq) in enumerate(batch):
            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq

    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)

    return head_batch, pos_batch

def get_data(args):
    seg_path = '/home/shuwen/data/data_preprocessing2/segment_labels/'
    attmat_path = '/home/shuwen/data/data_preprocessing2/record_attention_matrix/'
    cate_path = '/home/shuwen/data/data_preprocessing2/track_cate/'
    person_tracker_bbox = '../3d_pose2gaze/tracker_record_bbox/'
    person_battery_bbox = '../3d_pose2gaze/record_bbox/'
    save_path = '/home/shuwen/data/data_preprocessing2/segment_event_input/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    clips = os.listdir(seg_path)
    model = Atomic_node_only_lstm(args)
    args.cuda = True
    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
    model = load_best_checkpoint(model, path='.')  # todo: path here

    event_inputs = []
    event_labels = []

    for clip in clips:
        if not clip.split('.')[0] in event_seg_tracker:
            continue
        print(clip)
        with open(seg_path + clip, 'rb') as f:
            seg_dicts = pickle.load(f)
        with open(attmat_path + clip, 'rb') as f:
            attmat_obj = pickle.load(f)
        with open(cate_path + clip.split('.')[0] + '/' + clip, 'rb') as f:
            category = joblib.load(f)
        with open(person_tracker_bbox + clip, 'rb') as f:
            tracker_bbox = joblib.load(f)
        with open(person_battery_bbox + clip, 'rb') as f:
            battery_bbox = joblib.load(f)


        for seg in event_seg_tracker[clip.split('.')[0]]:
            if seg[1] - seg[0] < 5:
                continue
            else:
                test_seq = find_test_seq_over(attmat_obj, seg[0], seg[1],
                                              category, clip, tracker_bbox, battery_bbox)
                if len(test_seq) == 0:
                    continue

                test_set = mydataset_atomic(test_seq)
                test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic, batch_size=16,
                                                          shuffle=False)
                test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0,
                                                 'share': 0}

                test_results = test(test_loader, model, args)
                # print(test_results)
                event_input = merge_results(test_results)
                print(event_input)
                event_inputs.append(event_input)
                event_labels.append(seg[2])

        for seg in event_seg_battery[clip.split('.')[0]]:
            if seg[1] - seg[0] < 5:
                continue
            else:
                test_seq = find_test_seq_over(attmat_obj, seg[0], seg[1],
                                              category, clip, tracker_bbox, battery_bbox)
                if len(test_seq) == 0:
                    continue

                test_set = mydataset_atomic(test_seq)
                test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic, batch_size=16,
                                                          shuffle=False)
                test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0,
                                                 'share': 0}

                test_results = test(test_loader, model, args)
                # print(test_results)
                event_input = merge_results(test_results)

                event_inputs.append(event_input)
                event_labels.append(seg[2])

    with open('event_fine_tune_input.p', 'wb') as f:
        pickle.dump([event_inputs, event_labels], f)

def get_data_new_att(args):
    attmat_path = '../obj_oriented_event/attention_obj_name/'
    save_path = './ednet_event_input_new_att_1234/'
    # save_path = './atomic_output_new_att/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Atomic_node_only_lstm_first_view()
    args.cuda = True
    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
    model = load_best_checkpoint(model, path='./fine_tune_cptk_new_att_randomseed_1234/')  # todo: path here


    for clip in event_seg_tracker.keys():

        print(clip)
        event_inputs = []
        event_labels = []
        if not os.path.exists(attmat_path + clip + '.p'):
            continue
        test_results_dict = []
        with open(attmat_path + clip + '.p', 'rb') as f:
            attmat_obj = pickle.load(f)

        for seg in event_seg_tracker[clip.split('.')[0]]:
            if seg[1] - seg[0] < 5:
                continue
            else:
                test_seq = find_test_seq_over_new_att(attmat_obj, seg[0], seg[1],
                                              clip)


                test_set = mydataset_atomic_first_view_new_att(test_seq)
                test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic_first_view_new_att, batch_size=8,
                                                          shuffle=False)

                test_results = test(test_loader, model, args)

                frame_ids = [fid for _, fid, _ in test_seq]
                test_results_dict.append(list(zip(frame_ids, test_results)))
                # print(test_results)
                event_input = merge_results_new_att(test_results)

                event_inputs.append(event_input)
                event_labels.append(seg[2])
        # with open(save_path + clip + '.p', 'wb') as f:
        #     pickle.dump(test_results_dict, f)

        for seg in event_seg_battery[clip.split('.')[0]]:
            if seg[1] - seg[0] < 5:
                continue
            else:
                test_seq = find_test_seq_over_new_att(attmat_obj, seg[0], seg[1],
                                                      clip)

                test_set = mydataset_atomic_first_view_new_att(test_seq)
                test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic_first_view_new_att,
                                                          batch_size=8,
                                                          shuffle=False)

                test_results = test(test_loader, model, args)
                # print(test_results)
                event_input = merge_results_new_att(test_results)
                event_inputs.append(event_input)
                event_labels.append(seg[2])

        with open(save_path + clip + '.p', 'wb') as f:
            pickle.dump([event_inputs, event_labels], f)


def parse_arguments():
    project_name = 'test atomic event'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--project-name', default=project_name, help='project name')
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
    get_data_new_att(args)