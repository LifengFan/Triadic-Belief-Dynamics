import pickle
from metadata import *
from utils import *

def seg2frame(segs):
    frames = np.empty((1, 0))
    for seg in segs:
        start = seg[0]
        end = seg[1]
        frame = np.ones((1, end - start + 1))*seg[2]
        frames = np.hstack([frames, frame])
    assert frames.shape[1] == segs[-1][1] + 1
    return frames

def get_data(data_path):
    event_count = {0: 0, 1: 0, 2: 0}
    event0_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event1_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event2_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event_hash = {0: event0_count, 1: event1_count, 2: event2_count}
    mind_combination = {0:{}, 1:{}, 2:{}}
    labels = []
    frame_ids = []
    clips_id = []
    clips=listdir(data_path)
    mind_dict = {'m1':0, 'm2':1, 'm12':2, 'm21':3, 'mc':4}
    for clip in clips:
        if clip not in clips_with_gt_event:
            continue
        print(clip)
        event_segs = event_seg_battery[clip.split('.')[0]]
        event_by_frames = seg2frame(event_segs)
        with open(data_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        obj_names = obj_records.keys()
        for obj_name in obj_names:
            for frame_id in range(len(obj_records[obj_name])):
                event = event_by_frames[0][frame_id]
                obj_record = obj_records[obj_name][frame_id]
                key = [0, 0, 0, 0, 0]
                for mind_name in obj_record.keys():
                    if mind_name == 'mg':
                        continue
                    mind_change = obj_record[mind_name]['fluent']
                    event_hash[event][mind_name][mind_change] += 1
                    key[mind_dict[mind_name]] = mind_change

                key = tuple(key)
                if key in mind_combination[event]:
                    mind_combination[event][key] += 1
                else:
                    mind_combination[event][key] = 1

    print(event0_count)
    print(event1_count)
    print(event2_count)
    print(mind_combination)

    pass
    #         if model_type == 'single':
    #             vec_input, label_, frame_id = pickle.load(f)
    #             labels = labels + label_
    #             frame_ids = frame_ids + frame_id
    #             for i in range(len(label_)):
    #                 clips_id.append(clip)
    # assert len(labels) == len(frame_ids) == len(clips_id)
    # return labels, frame_ids, clips_id

def get_event(event_seg, frame_id):
    for seg in event_seg:
        if frame_id >= seg[0] and frame_id <= seg[1]:
            return seg[2]


def calculate_distribution(data_y, data_clip, data_frame_id, split):
    event_count = {0:0, 1:0, 2:0}
    event0_count = {'m1':{0:0, 1:0, 2:0, 3:0}, 'm2':{0:0, 1:0, 2:0, 3:0}, 'm12':{0:0, 1:0, 2:0, 3:0},
                    'm21':{0:0, 1:0, 2:0, 3:0}, 'mc':{0:0, 1:0, 2:0, 3:0}}
    event1_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event2_count = {'m1': {0: 0, 1: 0, 2: 0, 3: 0}, 'm2': {0: 0, 1: 0, 2: 0, 3: 0}, 'm12': {0: 0, 1: 0, 2: 0, 3: 0},
                    'm21': {0: 0, 1: 0, 2: 0, 3: 0}, 'mc': {0: 0, 1: 0, 2: 0, 3: 0}}
    event_hash = {0: event0_count, 1: event1_count, 2: event2_count}

    for clip_id, clip in enumerate(data_clip):
        print(clip)
        clip_name = clip.split('.')[0]
        frame_id = data_frame_id[clip_id]
        mind_change = np.array(data_y[clip_id]).reshape(-1)
        if not clip_name in event_seg_tracker:
            continue
        tracker_event_seg = event_seg_tracker[clip_name]
        event = get_event(tracker_event_seg, frame_id)
        assert event is not None
        event_count[event] += 1

        print(mind_change)
        labels_mc = np.argmax(mind_change[:4])
        labels_m21 = np.argmax(mind_change[4:8])
        labels_m12 = np.argmax(mind_change[8:12])
        labels_m1 = np.argmax(mind_change[12:16])
        labels_m2 = np.argmax(mind_change[16:20])

        event_hash[event]['mc'][labels_mc] += 1
        event_hash[event]['m21'][labels_m21] += 1
        event_hash[event]['m12'][labels_m12] += 1
        event_hash[event]['m1'][labels_m1] += 1
        event_hash[event]['m2'][labels_m2] += 1

    with open('./distribution_record/' + split + '.p', 'wb') as f:
        pickle.dump([event_count, event0_count, event1_count, event2_count], f)

    print(event0_count, event1_count, event2_count)

def calculate_between_mind_distribution(data_y, split):
    mind_combination = {}
    for mind_change in data_y:
        mind_change = np.array(mind_change).reshape(-1)
        labels_mc = np.argmax(mind_change[:4])
        labels_m21 = np.argmax(mind_change[4:8])
        labels_m12 = np.argmax(mind_change[8:12])
        labels_m1 = np.argmax(mind_change[12:16])
        labels_m2 = np.argmax(mind_change[16:20])
        key = (labels_m1, labels_m2, labels_m12, labels_m21, labels_mc)
        if key in mind_combination:
            mind_combination[key] += 1
        else:
            mind_combination[key] = 1

    print(mind_combination)
    with open('./distribution_record/' + split + '.p', 'wb') as f:
        pickle.dump(mind_combination, f)


if __name__ == '__main__':
    path=Path('home')
    data_path = path.reannotation_path
    get_data(data_path)
    # labels, frame_ids, clips_id = get_data(data_path)
    # calculate_distribution(labels, clips_id, frame_ids, 'event_mind_prob')
    # calculate_between_mind_distribution(labels, 'between_mind_prob')
