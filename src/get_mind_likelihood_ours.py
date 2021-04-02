from annotation_clean import *
from mind_model import *
from utils import *

def seg2frame_score(tracker_seg):
    event_frame = np.zeros((tracker_seg['cp'][-1], 3))
    for idx in range(len(tracker_seg['cp']) - 1):
        start = tracker_seg['cp'][idx]
        end = tracker_seg['cp'][idx + 1]
        # event = np.zeros((1, 3))
        # event[0][tracker_seg['event'][idx][0]] = 1
        event=tracker_seg['event_vec'][idx]

        for i in range(start, end):
            event_frame[i, :] = event
    return event_frame

def check_overlap_return_area(head_box, obj_curr):
    max_left = max(head_box[0], obj_curr[0])
    max_top = max(head_box[1], obj_curr[1])
    min_right = min(head_box[2], obj_curr[2])
    min_bottom = min(head_box[3], obj_curr[3])
    if (min_right - max_left) > 0 and (min_bottom - max_top) > 0:
        return (min_right - max_left)*(min_bottom - max_top)
    return -100

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
        if obj_candidate['lost'] == 1:
            continue
        candidate_bbox = [obj_candidate['x_min'], obj_candidate['y_min'], obj_candidate['x_max'], obj_candidate['y_max']]
        candidate_area = (obj_candidate['x_max'] - obj_candidate['x_min'])*(obj_candidate['y_max'] - obj_candidate['y_min'])
        overlap = check_overlap_return_area(obj_bbox, candidate_bbox)
        if overlap > max_overlap and overlap/obj_area < 1.2 and overlap/obj_area > 0.3 and overlap/candidate_area <1.2 and overlap/candidate_area>0.3:
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

def get_gt_data():

    path=Path('home')
    clip_path = path.home_path2+'BestTree_ours_0531/'
    reannotation_path = path.reannotation_path
    annotation_path = path.annotation_path
    color_img_path = path.img_path
    hog_path = path.feature_single
    cate_path = path.cate_path
    save_path = path.save_root+'/mind_likelihood_ours_0531/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clips = mind_test_clips
    with open(op.join(path.data_path, 'person_id.p'), 'rb') as f:
        person_ids = pickle.load(f)

    net = MindHog()
    net.load_state_dict(torch.load(path.mind_model_path))
    net.cuda()
    net.eval()

    obj_transforms = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])
    for clip in clips:
        if not os.path.exists(reannotation_path + clip):
            continue
        with open(op.join(path.home_path2, 'BestTree_ours_0531', clip), 'rb') as f:
            tree = pickle.load(f)

        print(clip)
        save_prefix = save_path + clip.split('.')[0] + '/'
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)
        with open(hog_path + clip, 'rb') as f:
            features = pickle.load(f)

        tracker_seg = tree['T']
        battery_seg = tree['B']

        tracker_events_by_frame = seg2frame_score(tracker_seg)
        battery_events_by_frame = seg2frame_score(battery_seg)
        assert  tracker_events_by_frame.shape == battery_events_by_frame.shape

        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)

        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
            p1_hog = features[1]
            p2_hog = features[2]
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
            p1_hog = features[2]
            p2_hog = features[1]

        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name","label"]
        obj_names = annt.name.unique()
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))

        for obj_name in obj_names:
            if os.path.exists(save_prefix + obj_name.split('/')[-1] + '.p'):
                continue
            if obj_name.startswith('P'):
                continue
            print(obj_name)

            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            mind_output_gt = []
            mind_output = []
            for frame_id in range(p1_events_by_frame.shape[0]):
                # curr_loc
                curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                curr_loc = get_grid_location(curr_df)
                print(curr_df)
                x_min = curr_df['x_min'].item()
                y_min = curr_df['y_min'].item()
                x_max = curr_df['x_max'].item()
                y_max = curr_df['y_max'].item()

                img = cv2.imread(img_names[frame_id])
                obj_record = obj_records[obj_name][frame_id]
                mind_output_gt.append(obj_record)

                # event
                p1_event = p1_events_by_frame[frame_id]
                p2_event = p2_events_by_frame[frame_id]

                if np.all(p1_event == np.array([0, 0, 0])):
                    p1_event = np.array([1/3., 1/3., 1/3.])

                if np.all(p2_event == np.array([0, 0, 0])):
                    p2_event = np.array([1/3., 1/3., 1/3.])

                # memory
                memory_dist = []
                indicator = []
                for mind_name in memory.keys():
                    if mind_name == 'mg':
                        continue
                    if frame_id == 0:
                        memory_dist.append(0)
                        indicator.append(0)
                    else:
                        memory_loc = memory[mind_name]['loc']
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

                # hog
                hog_input = np.hstack([p1_hog[frame_id][-162-10:-10], p2_hog[frame_id][-162-10:-10]])

                # obj_patch
                img_patch = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                # cv2.imshow('img', img_patch)
                # cv2.waitKey(20)
                obj_patch = transforms.ToPILImage()(img_patch)
                obj_patch = obj_transforms(obj_patch).numpy()

                event_input = np.hstack([p1_event, p2_event, memory_dist, indicator])

                event_input = torch.from_numpy(event_input).float().cuda().view((1, -1))
                hog_input = torch.from_numpy(hog_input).float().cuda().view((1, -1))

                obj_patch = torch.FloatTensor(obj_patch).cuda()
                obj_patch = obj_patch.view((1, 3, 224, 224))
                m1, m2, m12, m21, mc = net(event_input, obj_patch, hog_input)

                max_score, idx_mc = torch.max(mc, 1)
                max_score, idx_m21 = torch.max(m21, 1)
                max_score, idx_m12 = torch.max(m12, 1)
                max_score, idx_m1 = torch.max(m1, 1)
                max_score, idx_m2 = torch.max(m2, 1)
                print(idx_mc, idx_m1, idx_m2, idx_m12, idx_m21)
                m1 = torch.softmax(m1, dim = -1).data.cpu().numpy()
                m2 = torch.softmax(m2, dim=-1).data.cpu().numpy()
                m12 = torch.softmax(m12, dim=-1).data.cpu().numpy()
                m21 = torch.softmax(m21, dim=-1).data.cpu().numpy()
                mc = torch.softmax(mc, dim=-1).data.cpu().numpy()
                mind_output.append({'m1':m1, 'm2':m2, 'm12':m12, 'm21':m21, 'mc':mc, 'event':p1_event})

                memory = update_memory(memory, 'm1', idx_m1, curr_loc)
                memory = update_memory(memory, 'm2', idx_m2, curr_loc)
                memory = update_memory(memory, 'm12', idx_m12, curr_loc)
                memory = update_memory(memory, 'm21', idx_m21, curr_loc)
                memory = update_memory(memory, 'mc', idx_mc, curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

            assert len(mind_output) == len(mind_output_gt)
            if len(mind_output) > 0:
                with open(save_prefix + obj_name.split('/')[-1] + '.p', 'wb') as f:
                    pickle.dump([mind_output, mind_output_gt], f)


if __name__ == '__main__':

    get_gt_data()