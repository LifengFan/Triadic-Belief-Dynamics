import os
import pickle
import numpy as np
import sys
sys.path.append('/home/shuwen/data/Six-Minds-Project/data_processing_scripts/')
from metadata import *
import matplotlib.pyplot as plt

data_path = '/home/shuwen/data/data_preprocessing2/mind_retraining/'
model_type = 'single'

clips = os.listdir(data_path)
data = []
labels = []
frame_records = []
sep_data = {0:[], 1:[], 2:[], 3:[], 4:[]}
sep_label = {0:[], 1:[], 2:[], 3:[], 4:[]}
sep_clip = {0:[], 1:[], 2:[], 3:[]}
sep_frame_id = {0:[], 1:[], 2:[], 3:[]}
clip_dict = {}
for clip_id, clip in enumerate(clips):
    clip_dict[clip_id] = clip
    with open(data_path + clip, 'rb') as f:
        vec_input, label_, frame_id = pickle.load(f)
        data = data + vec_input
        labels = labels + label_
        for lid, label in enumerate(label_):
            label = label.reshape(-1)
            if label[0] == 1 or label[4] == 1 or label[8] == 1 or label[12] == 1 or label[16] == 1:
                sep_data[0].append(vec_input[lid])
                sep_label[0].append(label)
                sep_clip[0].append(clip_id)
                sep_frame_id[0].append(frame_id[lid])
            if label[1] == 1 or label[5] == 1 or label[9] ==  1 or label[13] == 1 or label[17] == 1:
                sep_data[1].append(vec_input[lid])
                sep_label[1].append(label)
                sep_clip[1].append(clip_id)
                sep_frame_id[1].append(frame_id[lid])
            if label[2] == 1 or label[6] == 1 or label[10] == 1 or label[14] == 1 or label[18] == 1:
                sep_data[2].append(vec_input[lid])
                sep_label[2].append(label)
                sep_clip[2].append(clip_id)
                sep_frame_id[2].append(frame_id[lid])
            if label[3] == 1 or label[7] == 1 or label[11] == 1 or label[15] == 1 or label[19] == 1:
                sep_data[3].append(vec_input[lid])
                sep_label[3].append(label)
                sep_clip[3].append(clip_id)
                sep_frame_id[3].append(frame_id[lid])

if model_type == 'single':
    test_xs, test_ys = np.empty((0, 22)), np.empty((0, 21))  # single
else:
    test_xs, test_ys = np.empty((0, 5, 20)), np.empty((0, 20)) #seq

sep_train_x, sep_train_y, sep_train_clip, sep_train_frame_id = {}, {}, {}, {}
test_clips = np.empty(0)
test_frame_ids = np.empty(0)
for i in sep_data.keys():
    if i == 4:
        continue
    data = sep_data[i]
    data = np.array(data)
    if model_type == 'single':
        data = data.reshape((-1, data.shape[-1]))
    label = sep_label[i]
    label = np.array(label)
    clip = sep_clip[i]
    clip = np.array(clip)
    frame_id = sep_frame_id[i]
    frame_id = np.array(frame_id)
    ratio = len(data) * 0.8
    ratio = int(ratio)
    indx = np.random.randint(0, len(data), ratio)
    flags = np.zeros(len(data))
    flags[indx] = 1
    train_x, train_y, train_clip, train_frame_id = data[flags == 1], label[flags == 1], clip[flags == 1], frame_id[flags == 1]
    test_x, test_y, test_clip, test_frame_id = data[flags == 0], label[flags == 0], clip[flags == 0], frame_id[flags == 0]
    sep_train_x[i] = train_x
    sep_train_y[i] = train_y
    sep_train_clip[i] = train_clip
    sep_train_frame_id[i] = train_frame_id

    test_xs = np.vstack([test_xs, test_x])
    test_ys = np.vstack([test_ys, test_y])
    test_clips = np.append(test_clips, test_clip)
    test_frame_ids = np.append(test_frame_ids, test_frame_id)

test_x, test_y = test_xs, test_ys
print(test_x.shape, test_y.shape)

max_sample = max(len(sep_train_x[0]), len(sep_train_x[1]), len(sep_train_x[2]), len(sep_train_x[3]))
if model_type == 'single':
    train_x, train_y = np.empty((0, 22)), np.empty((0, 21))  # single
else:
    train_x, train_y = np.empty((0, 5, 20)), np.empty((0, 20)) #seq

train_clip = np.empty(0)
train_frame_id = np.empty(0)
for i in sep_train_x.keys():
    repeat_time = max_sample/len(sep_train_x[i])
    # if i == 3:
    #     train_x = np.vstack([train_x, sep_train_x[i]])
    #     train_y = np.vstack([train_y, sep_train_y[i]])
    if not i==3:
        if model_type == 'single':
            train_x = np.vstack([train_x, np.tile(sep_train_x[i], (repeat_time*5, 1))])
        else:
            train_x = np.vstack([train_x, np.tile(sep_train_x[i], (repeat_time * 5, 1, 1))])
        train_clip = np.append(train_clip, np.tile(sep_train_clip[i], repeat_time*5))
        train_frame_id = np.append(train_frame_id, np.tile(sep_train_frame_id[i], repeat_time*5))
        train_y = np.vstack([train_y, np.tile(sep_train_y[i], (repeat_time*5, 1))])

if model_type == 'single':
    train_total = np.hstack([train_x, train_y])
else:
    train_total = np.hstack([train_x.reshape((-1, train_x.shape[1]*train_x.shape[2])), train_y])
train_total = np.hstack([train_total, train_clip.reshape((-1, 1)), train_frame_id.reshape((-1, 1))])
np.random.shuffle(train_total)
train_size = int(train_total.shape[0]*0.75)

if model_type == 'single':
    train_x, train_y, train_clip, train_frame_id = \
        train_total[:train_size, :22], train_total[:train_size, 22:-2], train_total[:train_size, -2], train_total[:train_size, -1]
    validate_x, validate_y, validate_clip, validate_frame_id = \
        train_total[train_size:, :22], train_total[train_size:, 22:-2], train_total[train_size:, -2], train_total[train_size:, -1]

else:
    train_x, train_y, train_clip, train_frame_id = \
        train_total[:train_size, :train_x.shape[1]*train_x.shape[2]].reshape((-1, train_x.shape[1], train_x.shape[2])), \
                       train_total[:train_size, train_x.shape[1]*train_x.shape[2]:-2], train_total[:train_size, -2], \
        train_total[:train_size, -1]
    validate_x, validate_y, validate_clip, validate_frame_id = \
        train_total[train_size:, :train_x.shape[1]*train_x.shape[2]].reshape((-1, train_x.shape[1], train_x.shape[2])), \
                             train_total[train_size:, train_x.shape[1]*train_x.shape[2]:-2], train_total[train_size:, -2], \
        train_total[train_size:, -1]


def get_event(event_seg, frame_id):
    for seg in event_seg:
        if frame_id >= seg[0] and frame_id <= seg[1]:
            return seg[2]

def plot_distri(dictionary, name, labels):
    print(dictionary.keys())
    values = dictionary.values()
    plt.bar(labels, values)
    plt.title(name)
    plt.show()


def calculate_distribution(data_y, data_clip, data_frame_id, split):
    event_count = {0:0, 1:0, 2:0}
    event0_count = {0:0, 1:0, 2:0, 3:0}
    event1_count = {0: 0, 1: 0, 2: 0, 3: 0}
    event2_count = {0: 0, 1: 0, 2: 0, 3: 0}
    event_hash = {0: event0_count, 1: event1_count, 2: event2_count}
    assert data_y.shape[0] == data_clip.shape[0] == data_frame_id.shape[0]
    for clip_id, clip in enumerate(data_clip):
        clip_name = clip_dict[clip].split('.')[0]
        frame_id = data_frame_id[clip_id]
        mind_change = data_y[clip_id]
        if not clip_name in event_seg_tracker:
            continue
        tracker_event_seg = event_seg_tracker[clip_name]
        event = get_event(tracker_event_seg, frame_id)
        assert event is not None
        event_count[event] += 1

        labels_mc = np.argmax(mind_change[:4])
        labels_m21 = np.argmax(mind_change[4:8])
        labels_m12 = np.argmax(mind_change[8:12])
        labels_m1 = np.argmax(mind_change[12:16])
        labels_m2 = np.argmax(mind_change[16:20])

        event_hash[event][labels_mc] += 1
        event_hash[event][labels_m21] += 1
        event_hash[event][labels_m12] += 1
        event_hash[event][labels_m1] += 1
        event_hash[event][labels_m2] += 1

    with open('./distribution_record/' + split + '.p', 'wb') as f:
        pickle.dump([event_count, event0_count, event1_count, event2_count], f)

    # plot_distri(event_count, split + ':event distribution', ['single', 'gaze follow', 'joint attention'])
    # plot_distri(event0_count, split + ':single: mind distribution', ['birth', 'disappear', 'update', 'null'])
    # plot_distri(event1_count, split + ':follow: mind distribution', ['birth', 'disappear', 'update', 'null'])
    # plot_distri(event2_count, split + ':joint: mind distribution', ['birth', 'disappear', 'update', 'null'])
    # plot_distri(event3_count, 'mutual: mind distribution')

print("train")
calculate_distribution(train_y, train_clip, train_frame_id, 'train')
print('validate')
calculate_distribution(validate_y, validate_clip, validate_frame_id, 'validate')
print('test')
calculate_distribution(test_y, test_clips, test_frame_ids, 'test')

def cal_transition_distribution():
    transition_record = {(0, 0):0, (0, 1):0, (0, 2):0, (1, 0):0, (1, 1):0, (1, 2):0, (2, 0):0, (2, 1):0, (2, 2):0}
    for clip in event_seg_tracker.keys():
        event_segs = event_seg_tracker[clip]
        for seg_id, event_seg in enumerate(event_segs):
            if seg_id == 0:
                continue
            key = (event_segs[seg_id - 1][2], event_segs[seg_id][2])
            if 3 in key:
                continue
            else:
                transition_record[key] += 1

    for clip in event_seg_battery.keys():
        event_segs = event_seg_battery[clip]
        for seg_id, event_seg in enumerate(event_segs):
            if seg_id == 0:
                continue
            key = (event_segs[seg_id - 1][2], event_segs[seg_id][2])
            if 3 in key:
                continue
            else:
                transition_record[key] += 1

    keys = map(str, transition_record.keys())
    with open('./distribution_record/transition.p', 'wb') as f:
        pickle.dump(transition_record, f)
    # plot_distri(transition_record, 'event transition', keys)

cal_transition_distribution()

# self.event_trans_table=pickle.load(open(os.path.join(args.stat_path, 'event_trans.p')))
# l1_sum=float(self.event_trans_table[(0,0)]+self.event_trans_table[(0,1)]+self.event_trans_table[(0,2)])
# l2_sum = float(self.event_trans_table[(1, 0)] + self.event_trans_table[(1, 1)] + self.event_trans_table[(1, 2)])
# l3_sum = float(self.event_trans_table[(2, 0)] + self.event_trans_table[(2, 1)] + self.event_trans_table[(2, 2)])
# self.event_trans_table[(0, 0)]=round(self.event_trans_table[(0, 0)]/l1_sum, 3)
# self.event_trans_table[(0, 1)]=round(self.event_trans_table[(0, 1)]/l1_sum,3)
# self.event_trans_table[(0, 2)] = round(self.event_trans_table[(0, 2)] / l1_sum, 3)
# self.event_trans_table[(1, 0)]=round(self.event_trans_table[(1, 0)] /l2_sum, 3)
# self.event_trans_table[(1, 1)]=round(self.event_trans_table[(1, 1)] /l2_sum, 3)
# self.event_trans_table[(1, 2)]=round(self.event_trans_table[(1, 2)] /l2_sum, 3)
# self.event_trans_table[(2, 0)]=round(self.event_trans_table[(2, 0)]/l3_sum, 3)
# self.event_trans_table[(2, 1)]=round(self.event_trans_table[(2, 1)]/l3_sum,3)
# self.event_trans_table[(2, 2)]=round(self.event_trans_table[(2, 2)]/l3_sum,3)
# with open(args.data_path+'event_trans_normalized.p', 'wb') as f:
#     pickle.dump(self.event_trans_table, f)