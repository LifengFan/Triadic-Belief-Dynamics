import joblib
from beam_search_m import *

args=parse_arguments()

for clip in clips_all:
    #
    # with open(args.tracker_bbox + clip, 'rb') as f:
    #     person_tracker_bbox = joblib.load(f)
    # with open('/media/lfan/HDD/NIPS20/data/tracker_record_bbox/'+clip,'wb') as f:
    #     pickle.dump(person_tracker_bbox,f)
    #
    # with open(args.battery_bbox + clip, 'rb') as f:
    #     person_battery_bbox = joblib.load(f)
    # with open('/media/lfan/HDD/NIPS20/data/record_bbox/'+clip,'wb') as f:
    #     pickle.dump(person_battery_bbox,f)
    #
    # with open(args.cate_path + clip.split('.')[0] + '/' + clip, 'rb') as f:
    #     category = joblib.load(f)
    # with open('/media/lfan/HDD/NIPS20/data/track_cate/'+clip,'wb') as f:
    #     pickle.dump(category,f)
    #
    # # feature
    # with open(os.path.join(args.data_path, 'feature_single', clip), 'rb') as f:
    #     feature_single = joblib.load(f)
    # with open('/media/lfan/HDD/NIPS20/data/feature_single/'+clip,'wb') as f:
    #     pickle.dump(feature_single,f)
    #
    # with open(os.path.join(args.data_path, 'feature_pair', clip), 'rb') as f:
    #     feature_pair = joblib.load(f)
    # with open('/media/lfan/HDD/NIPS20/data/feature_pair/'+clip,'wb') as f:
    #     pickle.dump(feature_pair,f)

    files=listdir(os.path.join(args.project_path, 'post_neighbor_smooth_newseq',  clip.split('.')[0]))

    if not os.path.exists(op.join(args.project_path2, 'data','post_neighbor_smooth_newseq', clip.split('.')[0])):
        os.makedirs(op.join(args.project_path2, 'data','post_neighbor_smooth_newseq', clip.split('.')[0]))

    for file in files:
        with open(op.join(args.project_path, 'post_neighbor_smooth_newseq',  clip.split('.')[0], file), 'rb') as f:
            obj_list=joblib.load(f)
        with open(op.join(args.project_path2, 'data','post_neighbor_smooth_newseq',  clip.split('.')[0], file), 'wb') as f:
            pickle.dump(obj_list, f)


