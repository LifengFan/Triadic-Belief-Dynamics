import os
import pickle
import numpy as np
import sys
sys.path.append('../data_processing_scripts/')
from metadata import *
import matplotlib.pyplot as plt

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