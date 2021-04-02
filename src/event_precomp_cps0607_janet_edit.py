from metadata import *
from utils import *
from overall_event_get_input import *
import os, os.path
from metric import *
from  Atomic_node_only_lstm_qls import  Atomic_node_only_lstm_first_view
from joblib import Parallel, delayed
import os.path as op
from train_fcnet import FCNet
from event_fine_tune import test, merge_results
import torchvision.transforms as transforms
import pandas as pd

def find_test_seq_over_new_att(attmat_obj, start_id, video_len, clip):
    test_seq = []
    while (start_id < video_len):
        obj_list_seq = {'P1': attmat_obj['P1'][start_id:start_id + 5], 'P2': attmat_obj['P2'][start_id:start_id + 5]}
        test_seq.append([clip, start_id, obj_list_seq])
        start_id = min(start_id + 5, video_len + 1)
    return test_seq

def test(test_loader, model, args):
    atomic_label={0:'single', 1:'mutual', 2:'avert', 3:'refer', 4:'follow', 5:'share'}
    test_results = []
    for i, (head_batch, pos_batch) in enumerate(test_loader):
        batch_size = head_batch.shape[0]
        if args.cuda:
            heads = (torch.autograd.Variable(head_batch)).cuda()
            poses = (torch.autograd.Variable(pos_batch)).cuda()
        with torch.set_grad_enabled(False):
            pred_atomic = model(heads, poses)
            for bid in range(batch_size):
                pred = torch.argmax(pred_atomic[bid, :], dim=0)
                test_results.append(pred.data.cpu().numpy())
    return test_results

def merge_results_new_att(test_results):
    labels = []
    freq = []
    counter = 0
    flag = 0
    for i in range(len(test_results)):
        atomic_label = test_results[i]
        if flag == 0:
            labels.append(atomic_label)
            freq.append(1)
            flag = 1
        else:
            if atomic_label == labels[counter]:
                freq[counter] += 1
            else:
                labels.append(atomic_label)
                freq.append(1)
                counter += 1
    return labels, freq

class mydataset_atomic_with_label_first_view_new_att(torch.utils.data.Dataset):
    def __init__(self, seq, args):
        self.seq = seq
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.seq_size = 5
        self.head_num=2
        self.node_num=4
        self.feature_dim = 162
        self.img_path=args.img_path
        self.bbox_path=args.bbox_path
        self.cate_path=args.cate_path
        self.feature_path=args.feature_single
        with open('./mind_change_classifier/person_id.p', 'rb') as f:
            self.person_ids = pickle.load(f)
        self.tracker_gt_path = args.tracker_gt_smooth
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
        with open(os.path.join(self.tracker_gt_path, clip), 'rb') as f:
            gazes = pickle.load(f, encoding='latin1')
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
                top = int(battery_img.shape[0] / 3 * 2)
                left = int(battery_img.shape[1] / 3)
                bottom = int(battery_img.shape[0])
                right = int(battery_img.shape[1] / 3 * 2)
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
                with open(obj_name, 'rb') as f:
                    obj_bboxs = pickle.load(f, encoding='latin1')
                obj_bbox = obj_bboxs[fid]
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

def normalize_event(event_input):
    new_input = []
    for eid, input in enumerate(event_input):
        input1, input2 = input
        count = 0
        event_dict = [0, 0, 0]
        for atomic_id, ato_event in enumerate(input1):
            freq = input2[atomic_id]
            event_dict[ato_event] += freq
            count += freq
        event_dict = np.array(event_dict)/float(count)
        assert abs(sum(event_dict) - 1) < 0.00001
        new_input.append(event_dict)
    return new_input

class EventScore(object):

    def __init__(self, atomic_net, event_net, args):

        self.init_cps_all=pickle.load(open(args.init_cps, 'rb'), encoding='latin1')
        self.args = args
        self.event_net=event_net
        self.atomic_net=atomic_net

    def run(self, clip):

        self.clip = clip
        self.clip_len = clips_len[self.clip]
        self.init_cps_T=self.init_cps_all[0][self.clip]
        self.init_cps_T.append(self.clip_len)
        self.init_cps_T=list(np.unique(self.init_cps_T))
        self.init_cps_B = self.init_cps_all[1][self.clip]
        self.init_cps_B.append(self.clip_len)
        self.init_cps_B = list(np.unique(self.init_cps_B))
        self.find_segs()

        with open(self.args.tracker_bbox + clip, 'rb') as f:
            self.person_tracker_bbox = pickle.load(f, encoding='latin1')
        with open(self.args.battery_bbox + clip, 'rb') as f:
            self.person_battery_bbox = pickle.load(f, encoding='latin1')

        # attmat
        with open(self.args.attmat_path + clip, 'rb') as f:
            self.attmat_obj = pickle.load(f, encoding='latin1')

        with open(self.args.cate_path + clip, 'rb') as f:
            self.category = pickle.load(f, encoding='latin1')

        if not op.exists(op.join(args.save_event_score, clip.split('.')[0])):
            os.makedirs(op.join(args.save_event_score, clip.split('.')[0]))

        if not op.exists(op.join(args.save_event_score, clip.split('.')[0], 'tracker')):
            os.makedirs(op.join(args.save_event_score, clip.split('.')[0], 'tracker'))

        if not op.exists(op.join(args.save_event_score, clip.split('.')[0], 'battery')):
            os.makedirs(op.join(args.save_event_score, clip.split('.')[0], 'battery'))

        for seg in self.segs_T:
            if os.path.exists(op.join(args.save_event_score, self.clip.split('.')[0], 'tracker','{}_{}.p'.format(seg[0], seg[1]))):
                continue
            outputs = self.event_score(seg[0], seg[1])
            with open(op.join(args.save_event_score, self.clip.split('.')[0], 'tracker','{}_{}.p'.format(seg[0], seg[1])), 'wb') as f:
                pickle.dump(outputs, f)
            print('[]/[] clip{} T seg {}'.format(self.clip, seg))

        for seg in self.segs_B:

            if os.path.exists(op.join(args.save_event_score, self.clip.split('.')[0], 'battery','{}_{}.p'.format(seg[0], seg[1]))):
                continue

            outputs = self.event_score(seg[0], seg[1])

            with open(op.join(args.save_event_score, self.clip.split('.')[0], 'battery','{}_{}.p'.format(seg[0], seg[1])), 'wb') as f:
                pickle.dump(outputs, f)
            print('[]/[] clip {} B seg {}'.format(self.clip, seg))

    def find_segs(self):

        self.segs_T=[]
        for id, start in enumerate(self.init_cps_T):
            for id2 in range(id+1, min(id+self.args.search_N_cp, len(self.init_cps_T))):
                end=self.init_cps_T[id2]
                self.segs_T.append([start,end])

        self.segs_B=[]
        for id, start in enumerate(self.init_cps_B):
            for id2 in range(id+1, min(id+self.args.search_N_cp, len(self.init_cps_B))):
                end=self.init_cps_B[id2]
                self.segs_B.append([start,end])

    def event_score(self, start, end):

        test_seq = find_test_seq_over_new_att(self.attmat_obj, start, end, self.clip,)
        if len(test_seq) == 0:
            return [[0.33, 0.33, 0.33]]
        else:
            test_set = mydataset_atomic_with_label_first_view_new_att(test_seq, self.args)
            test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic_first_view_new_att,batch_size=args.test_func_batch_size, shuffle=False)
            test_results = test(test_loader, self.atomic_net, self.args)
            input1, input2 = merge_results_new_att(test_results)
            input = normalize_event([[input1, input2]])
            if args.cuda:
                input1s = torch.tensor(input).float().cuda()
            else:
                input1s = torch.tensor(input).float()
            with torch.no_grad():
                outputs = self.event_net(input1s)
                outputs = torch.softmax(outputs,dim=-1)
            outputs = outputs.data.cpu().numpy()
            return outputs

def parse_arguments():

    parser=argparse.ArgumentParser(description='')
    # path
    home_path='/home/lfan/Dropbox/Projects/NIPS20/'
    home_path2='/media/lfan/HDD/NIPS20/'
    parser.add_argument('--project-path',default = home_path)
    parser.add_argument('--project-path2', default=home_path2)
    parser.add_argument('--data-path', default=home_path+'data/')
    parser.add_argument('--data-path2', default=home_path2 + 'data/')
    parser.add_argument('--img-path', default=home_path+'annotations/')
    parser.add_argument('--save-path', default='/media/lfan/HDD/NIPS20/Result/event_score_cps_new_01212021/')
    parser.add_argument('--init-cps', default='/media/lfan/HDD/NIPS20/data/init_cps/CPS_NEW_0601.p')
    parser.add_argument('--stat-path', default=home_path+'data/stat/')
    parser.add_argument('--attmat-path', default=home_path+'code/att_obj_id_w_raw_objs_01202021/')
    parser.add_argument('--cate-path', default=home_path2+'data/track_cate/')
    parser.add_argument('--tracker-bbox', default=home_path2+'data/tracker_record_bbox/')
    parser.add_argument('--battery-bbox', default=home_path2+'data/record_bbox/')
    parser.add_argument('--obj-bbox', default=home_path2+'data/post_neighbor_smooth_newseq/')
    parser.add_argument('--ednet-path', default=home_path+'code/fcnet_tuned_best_vid_split.pth')
    parser.add_argument('--atomic-path', default=home_path+'code/')
    parser.add_argument('--seg-label', default=home_path + 'data/segment_labels/')
    parser.add_argument('--feature-single', default=home_path2 + 'data/feature_single/')
    parser.add_argument('--save-event-score', default='/media/lfan/HDD/NIPS20/data/event_score_all_01212021/')
    parser.add_argument('--bbox-path', default=home_path+'reformat_annotation/')
    parser.add_argument('--tracker-gt-smooth', default=home_path2+'data/tracker_gt_smooth/')

    # parameter
    parser.add_argument('--lambda-1', default=1)
    parser.add_argument('--lambda-2', default=1)
    parser.add_argument('--lambda-3', default=1)
    parser.add_argument('--lambda-4', default=1)
    parser.add_argument('--lambda-5', default=1)
    parser.add_argument('--lambda-6', default=1)
    parser.add_argument('--beta-1', default=1)
    parser.add_argument('--beta-2', default=1)
    parser.add_argument('--beta-3', default=1)
    parser.add_argument('--search-N-cp', default=6)
    parser.add_argument('--topN', default=1)
    parser.add_argument('--hist-bin', default=10)
    parser.add_argument('--seg-alpha', default=10) #todo: check the alpha here!

    # others
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--ip', default='192.168.1.17')
    parser.add_argument('--port', default=1234)
    parser.add_argument('--resume',default=False) # to resume from the last point
    parser.add_argument('--test-func-batch-size', default=1)
    return parser.parse_args()

def load_best_checkpoint(args,model,optimizer,path):
    if path:
       checkpoint_dir=path
       best_model_file=os.path.join(checkpoint_dir,'atomic_best_new_att.pth')
       if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
       if os.path.isfile(best_model_file):
            print("====> loading best model {}".format(best_model_file))
            checkpoint=torch.load(best_model_file, encoding='latin1')
            args.start_epoch=checkpoint['epoch']
            best_epoch_error=checkpoint['best_epoch_acc']
            try:
                avg_epoch_error=checkpoint['avg_epoch_acc']
            except KeyError:
                avg_epoch_error=np.inf
            model_dict = model.state_dict()
            pretrained_model = checkpoint['state_dict']
            pretrained_dict = {}
            for k, v in pretrained_model.items():
                if k[len('module.'):] in model_dict:
                    pretrained_dict[k[len('module.'):]] = v
                elif k in model_dict:
                    pretrained_dict[k]=v
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.cuda:
                model.cuda()
            print("===> loaded best model {} (epoch {})".format(best_model_file,checkpoint['epoch']))
            return args, best_epoch_error, avg_epoch_error, model, optimizer
       else:
           print('===> no best model found at {}'.format(best_model_file))
    else:
        return None

if __name__ == "__main__":

    args = parse_arguments()

    atomic_net = Atomic_node_only_lstm_first_view()
    optimizer = torch.optim.Adam(atomic_net.parameters())
    load_best_checkpoint(args,atomic_net,optimizer, path=args.atomic_path)
    if args.cuda and torch.cuda.is_available():
        atomic_net.cuda()
    atomic_net.eval()
    event_net=FCNet()

    event_net.load_state_dict(torch.load(args.ednet_path))
    if args.cuda and torch.cuda.is_available():
        event_net.cuda()
    event_net.eval()

    event_score = EventScore(atomic_net, event_net, args)

    clips_gt_annot=[]
    for clip in mind_test_clips:
        if not op.exists(args.attmat_path + clip):
            continue
        else:
            clips_gt_annot.append(clip)

    print(clips_gt_annot)

    Parallel(n_jobs=1)(delayed(event_score.run)(clip) for _, clip in enumerate(clips_gt_annot))

