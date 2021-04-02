import os
import pickle
from Atomic_node_only_lstm_qls import Atomic_node_only_lstm
import argparse
from test_communication_event import *
import torch.nn as nn
import torch.nn.functional as F

def find_test_seq_over(attmat_obj, start_id, video_len, category, clip, tracker_bbox, battery_bbox):
    slide_size = 5
    test_seq = []
    attmat, obj_list = attmat_obj
    while (start_id < video_len):
        obj_list_seq = check_atom(attmat, start_id, video_len, obj_list, category, clip, tracker_bbox, battery_bbox)
        if obj_list_seq:
            # print([clip, start_id, obj_list_seq, obj_list])
            test_seq.append([clip, start_id, obj_list_seq, obj_list])
            start_id = min(start_id + 5, video_len + 1)
        else:
            start_id += slide_size
    return test_seq

def merge_results(test_results):
    labels = []
    freq = []
    counter = 0
    for i in range(len(test_results)):
        index_batch = test_results[i][0]
        for batch_id, batch in enumerate(index_batch):
            atomic_label = test_results[i][batch_id + 2]
            if batch_id == 0:
                labels.append(atomic_label)
                freq.append(1)
            else:
                if atomic_label == labels[counter]:
                    freq[counter] += 1
                else:
                    labels.append(atomic_label)
                    freq.append(1)
                    counter += 1
    return labels, freq



def main(args):
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
    # clips = ['test_boelter2_2.p']
    for clip in clips:
        # if os.path.exists(save_path + clip):
        #     continue
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
        battery_seg_event_label = []
        tracker_seg_event_label = []
        for seg_dict in seg_dicts:
            first_level_seg = seg_dict['first_level']
            if first_level_seg[1] - first_level_seg[0] < 5:
                battery_seg_event_label.append([first_level_seg[0], first_level_seg[1],'NA'])
                tracker_seg_event_label.append([first_level_seg[0], first_level_seg[1],'NA'])
            else:
                start_id = first_level_seg[0]
                second_level_seg_tracker = seg_dict['second_level_tracker']
                if type(second_level_seg_tracker[0]) == int:
                    second_level_seg_tracker = [second_level_seg_tracker]
                for second_seg in second_level_seg_tracker:
                    test_seq = find_test_seq_over(attmat_obj, start_id + second_seg[0], start_id + second_seg[1], category, clip, tracker_bbox, battery_bbox)
                    if len(test_seq) == 0:
                        print('tra', start_id + second_seg[0], start_id + second_seg[1], second_seg[1] - second_seg[0])
                        tracker_seg_event_label.append(
                            [second_seg[0] + start_id, second_seg[1] + start_id, 'NA'])
                        continue
                    test_set = mydataset_atomic(test_seq)
                    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic, batch_size=16,
                                                              shuffle=False)
                    test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0,
                                                     'share': 0}

                    test_results = test(test_loader, model, args)
                    # print(test_results)
                    event_input = merge_results(test_results)
                    tracker_seg_event_label.append([second_seg[0] + start_id, second_seg[1] + start_id, event_input])

                second_level_seg_battery = seg_dict['second_level_battery']
                if type(second_level_seg_battery[0]) == int:
                    second_level_seg_battery = [second_level_seg_battery]
                for second_seg in second_level_seg_battery:
                    test_seq = find_test_seq_over(attmat_obj, start_id + second_seg[0], start_id + second_seg[1], category, clip, tracker_bbox, battery_bbox)
                    if len(test_seq) == 0:
                        print('bat', start_id + second_seg[0], start_id + second_seg[1], second_seg[1] - second_seg[0])
                        battery_seg_event_label.append(
                            [second_seg[0] + start_id, second_seg[1] + start_id, 'NA'])
                        continue
                    test_set = mydataset_atomic(test_seq)
                    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic, batch_size=16,
                                                              shuffle=False)
                    test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0,
                                                     'share': 0}

                    test_results = test(test_loader, model, args)
                    event_input = merge_results(test_results)
                    battery_seg_event_label.append([second_seg[0] + start_id, second_seg[1] + start_id, event_input])
        event_input_dict = {'battery':battery_seg_event_label, 'tracker':tracker_seg_event_label}
        with open(save_path + clip, 'wb') as f:
            pickle.dump(event_input_dict, f)

# class EDNet(nn.Module):
#     def __init__(self):
#         super(EDNet, self).__init__()
#         self.encoder_1 = nn.Linear(50, 50)
#         self.encoder_2 = nn.Linear(50, 50)
#         self.decoder = nn.Linear(100, 5)
#
#     def forward(self, x_1, x_2):
#         latent_1 = F.dropout(F.relu(self.encoder_1(x_1)), 0.8)
#         latent_2 = F.dropout(F.relu(self.encoder_2(x_2)), 0.8)
#         x = self.decoder(torch.cat((latent_1, latent_2), 1))
#         return x

class EDNet(nn.Module):
    def __init__(self):
        super(EDNet, self).__init__()
        self.encoder_1 = nn.Linear(50, 50)
        self.encoder_2 = nn.Linear(50, 50)
        self.decoder_1 = nn.Linear(100, 50)
        self.decoder_2=nn.Linear(50,3)


    def forward(self, x_1, x_2):
        latent_1 = F.relu(self.encoder_1(x_1))
        latent_2 = F.relu(self.encoder_2(x_2))
        x = F.relu(self.decoder_1(torch.cat((latent_1, latent_2), 1)))
        #x = torch.cat((latent_1, latent_2), 1)
        x=self.decoder_2(x)

        return x
def event_output():
    input_path = '/home/shuwen/data/data_preprocessing2/segment_event_input/'
    img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    save_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    CLASS = {0:'SingleGaze', 1:'MutualGaze', 2:'AvertGaze', 3:'GazeFollow', 4:'JointAtt', -1:'NA'}
    clips = os.listdir(input_path)
    net = EDNet()
    net.load_state_dict(torch.load('model_event.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for clip in clips:
        print(clip)
        img_names = sorted(glob.glob(img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        with open(input_path + clip, 'rb') as f:
            event_inputs = pickle.load(f)
        results_dict = {}
        # battery
        battery_inputs = event_inputs['battery']
        input1s, input2s = np.empty((0, 50)), np.empty((0, 50))
        ignore_input_ids = []
        record_input_ids = []
        for input_id, input in enumerate(battery_inputs):
            if input[2] == 'NA':
                ignore_input_ids.append(input_id)
                continue
            input1, input2 = input[2]
            input1_pad = np.zeros((1, 50))
            input2_pad = np.zeros((1, 50))
            for i in range(len(input1)):
                input1_pad[0, i] = input1[i]
            for i in range(len(input2)):
                input2_pad[0, i] = input2[i]
            input1s = np.vstack([input1s, input1_pad])
            input2s = np.vstack([input2s, input2_pad])
            record_input_ids.append(input_id)
        input1s = torch.tensor(input1s).float().cuda()
        input2s = torch.tensor(input2s).float().cuda()
        outputs = net(input1s, input2s)
        assert outputs.shape[0] > 0
        max_score, idx = torch.max(outputs, 1)
        idx = idx.cpu().numpy()
        outputs = outputs.data.cpu().numpy()

        results = []
        for input_id, input in enumerate(battery_inputs):
            if input_id in ignore_input_ids:
                results.append([input[:2], 'NA'])
            else:
                ind = record_input_ids.index(input_id)
                results.append([input[:2], outputs[ind]])
        results_dict['battery'] = results

        # tracker
        tracker_inputs = event_inputs['tracker']
        input1s, input2s = np.empty((0, 50)), np.empty((0, 50))
        ignore_input_ids = []
        record_input_ids = []
        for input_id, input in enumerate(tracker_inputs):
            if input[2] == 'NA':
                ignore_input_ids.append(input_id)
                continue
            input1, input2 = input[2]
            input1_pad = np.zeros((1, 50))
            input2_pad = np.zeros((1, 50))
            for i in range(len(input1)):
                input1_pad[0, i] = input1[i]
            for i in range(len(input2)):
                input2_pad[0, i] = input2[i]
            input1s = np.vstack([input1s, input1_pad])
            input2s = np.vstack([input2s, input2_pad])
            record_input_ids.append(input_id)
        input1s = torch.tensor(input1s).float().cuda()
        input2s = torch.tensor(input2s).float().cuda()
        outputs = net(input1s, input2s)
        assert outputs.shape[0] > 0
        max_score, idx = torch.max(outputs, 1)
        idx = idx.cpu().numpy()
        outputs = outputs.data.cpu().numpy()

        results = []
        for input_id, input in enumerate(tracker_inputs):
            if input_id in ignore_input_ids:
                results.append([input[:2], 'NA'])
            else:
                ind = record_input_ids.index(input_id)
                results.append([input[:2], outputs[ind]])
        results_dict['tracker'] = results

        with open(save_path + clip, 'wb') as f:
            pickle.dump(results_dict, f)

        # labels = np.zeros(len(img_names))
        # segment_labels = np.zeros(len(img_names))
        # for input_id, input in enumerate(battery_inputs):
        #     if input_id in ignore_input_ids:
        #         if input[1] + 1 >= len(img_names):
        #             end = len(img_names)
        #         else:
        #             end = input[1] + 1
        #         # print(input[0], end)
        #         labels[input[0]:end] = -1
        #         segment_labels[input[0]:end] = input_id
        #     else:
        #         ind = record_input_ids.index(input_id)
        #         if input[1] + 1 >= len(img_names):
        #             end = len(img_names)
        #         else:
        #             end = input[1] + 1
        #         # print(input[0], end)
        #         labels[input[0]:end] = idx[ind]
        #         segment_labels[input[0]:end] = input_id

        # for frame_id, img_name in enumerate(img_names):
        #     img = cv2.imread(img_name)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     org = (50, 50)
        #     fontScale = 1
        #     color = (255, 0, 0)
        #     thickness = 2
        #     cv2.putText(img, 'seg:{}'.format(segment_labels[frame_id]) + ' ' + CLASS[int(labels[frame_id])],
        #                 (org[0], org[1]), font,
        #                 fontScale, color, thickness, cv2.LINE_AA)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(20)


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
    main(args)
    # event_output()
