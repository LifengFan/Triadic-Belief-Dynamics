import os
from Atomic_node_only_lstm_517 import Atomic_node_only_lstm, Atomic_node_only_lstm_first_view
import torch
import utils_atomic
import pickle
import joblib
import sys
from metadata import *
from overall_event_get_input import *
import argparse

def get_data(args):

    seg_path = args.seg_label #'/home/shuwen/data/data_preprocessing2/segment_labels/'
    attmat_path = args.attmat_path #'/home/shuwen/data/data_preprocessing2/record_attention_matrix/'
    cate_path = args.cate_path #'/home/shuwen/data/data_preprocessing2/track_cate/'
    person_tracker_bbox = args.tracker_bbox #'../3d_pose2gaze/tracker_record_bbox/'
    person_battery_bbox = args.battery_bbox #'../3d_pose2gaze/record_bbox/'
    save_path = args.save_path #'/home/shuwen/data/data_preprocessing2/segment_event_input/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    clips = os.listdir(seg_path)

    model = Atomic_node_only_lstm_first_view()
    args.cuda = True
    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()
    _, _, _, model, _ = utils_atomic.load_best_checkpoint(args, model, None, path=op.join(args.project_path, 'code', 'model_best_event.pth'))  # todo: path here

    event_inputs = []
    event_labels = []

    for clip in clips:
        if not clip.split('.')[0] in event_seg_tracker:
            continue
        print(clip)
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

        for seg in event_seg_tracker[clip.split('.')[0]]:
            if seg[1] - seg[0] < 5:
                continue
            else:
                test_seq = find_test_seq_over(attmat_obj, seg[0], seg[1],
                                              category, clip, tracker_bbox, battery_bbox)
                if len(test_seq) == 0:
                    continue

                test_set = mydataset_atomic_with_label_first_view(test_seq, args)
                test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic_first_view, batch_size=16,
                                                          shuffle=False)
                test_loader.dataset.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0,
                                                 'share': 0}

                test_results = test(test_loader, model, args)
                # print(test_results)
                event_input = merge_results(test_results)

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

                test_set = mydataset_atomic_with_label_first_view(test_seq,args)
                test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic_first_view, batch_size=16,
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