import os
import joblib
import numpy as np
import pandas as pd
import glob
import cv2


class BoxSmooth:
    def __init__(self):
        self.box_path = './deep_sort/result/'
        self.save_prefix = './box_reid/'
        self.img_path = './to_track/'
        self.mask_path = './resort_detected_imgs/kinect/masks/'
        self.save_smooth_prefix = './post_box_reid/'
        self.save_frame_prefix = './frame_record/'
        self.save_neighbor_prefix = './neihbor_record/'
        self.save_cat_prefix = './track_cate/'
        self.save_neighbor_smooth_prefix = './neihbor_smooth/'
        self.neighbor_smooth_newseq = './neighbor_smooth_newseq/'
        self.post_smooth_newseq = './post_neighbor_smooth_newseq/'
        self.obj_refer = {'handbag': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'suitcase': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'backpack': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'bowl': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'apple': ['apple', 'sports ball'],
                          'sports ball': ['apple', 'sports ball'],
                          'cup': ['bottle', 'wine glass'],
                          'bottle': ['cup', 'wine glass'],
                          'wine glass': ['cup', 'bottle'],
                          'book': ['tv'],
                          'tv': ['book'],
                          'banana': ['sandwich', 'teddy bear', 'donut'],
                          'sandwich': ['banana', 'teddy bear', 'donut'],
                          'donut': ['banana', 'teddy bear', 'sandwich'],
                          'teddy bear': ['sandwich', 'banana', 'donut']}
        self.obj_refer_b4 = {'handbag': ['handbag', 'suitcase', 'backpack', 'bowl'],
                             'suitcase': ['handbag', 'suitcase', 'backpack', 'bowl'],
                             'backpack': ['handbag', 'suitcase', 'backpack', 'bowl'],
                             'bowl': ['handbag', 'suitcase', 'backpack', 'bowl'],
                             'apple': ['apple', 'sports ball'],
                             'sports ball': ['apple', 'sports ball'],
                             'cup': ['bottle', 'wine glass', 'cell phone'],
                             'bottle': ['cup', 'wine glass', 'cell phone'],
                             'cell phone': ['cup', 'wine glass', 'bottle'],
                             'wine glass': ['cup', 'bottle', 'cell phone'],
                             'book': ['tv'],
                             'tv': ['book'],
                             'banana': ['sandwich', 'teddy bear', 'donut'],
                             'sandwich': ['banana', 'teddy bear', 'donut'],
                             'donut': ['banana', 'teddy bear', 'sandwich'],
                             'teddy bear': ['sandwich', 'banana', 'donut']}

    def box_rename(self):
        clips = sorted(glob.glob(self.box_path + '*.txt'))
        for clip in clips:
            print(clip)
            save_path = self.save_prefix + clip.split('/')[-1].split('.')[0]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue
            boxes = pd.read_csv(clip, sep=",", header=None)
            boxes.columns = ['frame', 'obj', 'box1', 'box2', 'box3', 'box4', 'box5', '3d1', '3d2', '3d3']
            frame_length = max(boxes['frame'])
            obj_ids = boxes['obj'].unique()
            sep_boxes = dict()
            for obj_id in obj_ids:
                sep_boxes[obj_id] = []
            for i in range(frame_length):
                print(i)
                for obj_id in obj_ids:
                    sel_box = boxes.loc[(boxes['frame'] == i + 1) & (boxes['obj'] == obj_id)]
                    if len(sel_box) > 0:
                        sel_box = sel_box.values.tolist()[0]
                        sep_boxes[obj_id].append([sel_box[2], sel_box[3], sel_box[4], sel_box[5], sel_box[6]])
                    else:
                        sep_boxes[obj_id].append([0, 0, 0, 0, 0])

            for obj_id in obj_ids:
                with open(save_path + '/' + str(obj_id) + '.p', 'wb') as f:
                    joblib.dump(sep_boxes[obj_id], f)

    def box_frame_record(self):
        clips = os.listdir(self.save_smooth_prefix)
        for clip in clips:
            print(clip)
            save_path = self.save_frame_prefix + clip
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue
            obj_names = sorted(glob.glob(os.path.join(self.save_smooth_prefix, clip) + '/*.p'))
            record_frames = dict()
            fm = open(save_path + '/' + clip + '.txt', 'w')
            for obj_name in obj_names:
                with open(obj_name, 'rb') as f_:
                    frames = joblib.load(f_)
                start_frames = []
                end_frames = []
                start_id = None
                start_flag = 0
                appear_frame = len(frames) - 1
                for frame_id, frame in enumerate(frames):
                    if start_flag == 0:
                        if np.mean(frame) > 0:
                            start_flag = 1
                            appear_frame = frame_id
                    else:
                        if np.mean(frame) == 0:
                            if start_id == None:
                                start_frames.append(frame_id)
                                start_id = frame_id
                        elif not start_id == None:
                            end_frames.append(frame_id)
                            start_id = None
                        if not start_id == None and frame_id == len(frames) - 1:
                            end_frames.append(frame_id)
                assert len(start_frames) == len(end_frames)
                if len(start_frames) > 0:
                    fm.write(obj_name + ',' + str(appear_frame) + ',' + str(start_frames[0]) + ',' + str(
                        end_frames[0]) + '\n')
                else:
                    fm.write(obj_name + ',' + str(appear_frame) + ',' + str(len(frames) - 1) + ',' + str(
                        len(frames) - 1) + '\n')
                record_frames[obj_name] = [start_frames, end_frames]
            fm.close()
            with open(save_path + '/' + clip + '.p', 'wb') as f:
                joblib.dump(record_frames, f)

    def box_category_record(self):
        clips = os.listdir(self.save_prefix)
        dete_prefix = './resort_detected_imgs/kinect/objs/'

        for clip in clips:
            # if not clip == "test_94342_24":
            #     continue
            print(clip)
            save_path = self.save_cat_prefix + clip
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue
            obj_names = sorted(glob.glob(os.path.join(self.save_smooth_prefix, clip) + '/*.p'))
            with open(dete_prefix + clip + '.p', 'rb') as f:
                dete_frames = joblib.load(f)
            cate = dict()
            with open("./to_track/" + clip + '/seqinfo.ini', 'rb') as f:
                infos = f.readlines()
            image_dir = infos[2].split('=')[1][:-1]
            kinect_img_names = sorted(glob.glob(image_dir + '/*.jpg'))
            # kinect_img_names = sorted(glob.glob('./to_track/' + clip + '/img1/*.jpg'))
            for obj_name in obj_names:
                with open(obj_name, 'rb') as f_:
                    frames = joblib.load(f_)
                for frame_id, frame in enumerate(frames):
                    if np.mean(frame) > 0:
                        dist_min = 100
                        cat_min = None
                        sub_min = None
                        obj_min = None
                        # box_min = None

                        dete_frame = dete_frames[frame_id]
                        img = cv2.imread(kinect_img_names[frame_id])
                        for obj_id, obj in enumerate(dete_frame):
                            for sub_id, sub_obj in enumerate(obj[1]):
                                dist = np.linalg.norm(
                                    np.array([frame[0], frame[1], frame[0] + frame[2], frame[1] + frame[3]]) - np.array(
                                        sub_obj[:-1]))
                                if dist < dist_min:
                                    dist_min = dist
                                    cat_min = obj[0]
                                    sub_min = sub_id
                                    obj_min = obj_id
                        if not cat_min == None:
                            cate[obj_name] = [cat_min, obj_min, sub_min]
                            break
                        cate[obj_name] = [cat_min, obj_min, sub_min]
                        # cv2.rectangle(img, (int(frame[0]), int(frame[1])), (int(frame[0] + frame[2]), int(frame[1] + frame[3])), (255,0,255), thickness=5)
                        # cv2.rectangle(img, (int(box_min[0]), int(box_min[1])), (int(box_min[2]), int(box_min[3])), (255,0,0), thickness=5)
                        # cv2.imshow(cat_min, img)
                        # cv2.waitKey(20)
                        # raw_input("Enter")
            with open(save_path + '/' + clip + '.p', 'wb') as f:
                joblib.dump(cate, f)

    def point2screen(self, points):
        K = [607.13232421875, 0.0, 638.6468505859375, 0.0, 607.1067504882812, 367.1607360839844, 0.0, 0.0, 1.0]
        K = np.reshape(np.array(K), [3, 3])
        rot_points = np.array(points)
        points_camera = rot_points.reshape(3, 1)
        project_matrix = np.array(K).reshape(3, 3)
        points_prj = project_matrix.dot(points_camera)
        points_prj = points_prj.transpose()
        if not points_prj[:, 2][0] == 0.0:
            points_prj[:, 0] = points_prj[:, 0] / points_prj[:, 2]
            points_prj[:, 1] = points_prj[:, 1] / points_prj[:, 2]
        points_screen = points_prj[:, :2]
        assert points_screen.shape == (1, 2)
        points_screen = points_screen.reshape(-1)
        return points_screen

    def extract_feature(self, cropped_img):
        hists = np.zeros((np.arange(0, 256).shape[0] - 1) * 3)
        hists[:255] = np.histogram(cropped_img[:, :, 0], bins=np.arange(0, 256), density=True)[0]
        hists[255:255 * 2] = np.histogram(cropped_img[:, :, 1], bins=np.arange(0, 256), density=True)[0]
        hists[255 * 2:255 * 3] = np.histogram(cropped_img[:, :, 2], bins=np.arange(0, 256), density=True)[0]

        return hists

    def switch_key(self, parentness, obj_candidate, obj_name):
        if obj_candidate in parentness.keys():
            if int(obj_name.split('/')[-1].split('.')[0]) < int(parentness[obj_candidate].split('/')[-1].split('.')[0]):
                self.switch_key(parentness, parentness[obj_candidate], obj_name)
                # parentness[parentness[obj_candidate]] = obj_name
            else:
                parentness[obj_name] = parentness[obj_candidate]
                parentness[obj_candidate] = obj_name

        else:
            parentness[obj_candidate] = obj_name

    def box_find_neighbor(self):
        kinect_skeles_path = './post_images/kinect/'
        video_folders = os.listdir(kinect_skeles_path)
        for video_folder in video_folders:
            clips = os.listdir(kinect_skeles_path + video_folder)
            for clip in clips:
                # if not clip == "test_boelter4_3": #and not clip == "test_94342_18":
                #     continue
                print(clip)

                if len(clip.split('_')) > 1 and clip.split('_')[1] == 'boelter4':
                    obj_refer = self.obj_refer_b4
                else:
                    obj_refer = self.obj_refer

                save_path = self.save_neighbor_prefix + clip

                obj_names = sorted(glob.glob(os.path.join(self.save_smooth_prefix, clip) + '/*.p'))

                if not os.path.exists(self.save_frame_prefix + clip + '/' + clip + '.txt'):
                    continue
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    continue

                df = pd.read_csv(self.save_frame_prefix + clip + '/' + clip + '.txt', sep=",", header=None)
                df.columns = ['obj', 'appear_frame', 'start_frame', 'end_frame']

                # with open(os.path.join(kinect_skeles_path, video_folder, clip) + '/skele1.p', 'rb') as f:
                #     skele1 = joblib.load(f)
                # with open(os.path.join(kinect_skeles_path, video_folder, clip) + '/skele2.p', 'rb') as f:
                #     skele2 = joblib.load(f)

                kinect_img_names = sorted(glob.glob(os.path.join(kinect_skeles_path, video_folder, clip) + '/*.jpg'))
                with open(self.save_cat_prefix + clip + '/' + clip + '.p', 'rb') as f:
                    cate = joblib.load(f)

                # with open(self.mask_path + clip + '.p', 'rb') as f:
                #     mask_frames = joblib.load(f)

                parentness = dict()
                for obj_name in obj_names:
                    print('obj', obj_name, cate[obj_name])
                    with open(obj_name, 'rb') as f:
                        frames = joblib.load(f)
                    obj_record_frame = df.loc[df['obj'] == obj_name].values.tolist()
                    if len(obj_record_frame) == 0 or obj_record_frame[0][1] == obj_record_frame[0][2]:
                        continue
                    start_frame = obj_record_frame[0][2]
                    kinect_img = cv2.imread(kinect_img_names[start_frame - 1])
                    frame = frames[start_frame - 1]
                    frame[frame < 0] = 0
                    obj_img = kinect_img[int(frame[1]):int(frame[1] + frame[3]), int(frame[0]):int(frame[0] + frame[2]),
                              :]
                    obj_feature = self.extract_feature(obj_img)
                    obj_cate = cate[obj_name][0]
                    obj_candidates = df.loc[df['appear_frame'] >= start_frame]['obj']

                    feature_dists = []
                    feature_infos = []
                    # print('./post_box_reid/test_94342_7/192.p', cate['./post_box_reid/test_94342_7/192.p'])
                    for obj_candidate in obj_candidates:
                        print('can:', obj_candidate, cate[obj_candidate])

                        if (obj_cate == cate[obj_candidate][0]) or (
                                obj_cate in obj_refer.keys() and cate[obj_candidate][0] in obj_refer[obj_cate]):
                            with open(obj_candidate, 'rb') as f:
                                candidate_frames = joblib.load(f)
                            candidate_appear_frame = df.loc[df['obj'] == obj_candidate]['appear_frame'].values.tolist()[
                                0]
                            candidate_frame = candidate_frames[candidate_appear_frame]
                            print(candidate_frame)
                            candidate_frame[candidate_frame < 0] = 0
                            # cv2.imshow('can:', candidate_img)
                            # cv2.imshow('obj:' , obj_img)
                            # cv2.imwrite('./can.jpg', candidate_img)
                            # print(cate[obj_candidate][0])
                            # cv2.waitKey(20)
                            # raw_input("Enter")

                            candi_img = cv2.imread(kinect_img_names[candidate_appear_frame])
                            candidate_img = candi_img[
                                            int(candidate_frame[1]):int(candidate_frame[1] + candidate_frame[3]),
                                            int(candidate_frame[0]):int(candidate_frame[0] + candidate_frame[2]), :]
                            candidate_feature = self.extract_feature(candidate_img)
                            feature_dist = np.linalg.norm(obj_feature - candidate_feature)
                            cv2.imshow('can:', candidate_img)
                            cv2.imshow('obj:', obj_img)
                            print(feature_dist)
                            cv2.waitKey(20)
                            # raw_input("Enter")
                            # if feature_dist < 0.13:
                            #     # obj_candidate = feature_infos[min_idx]
                            #     print(obj_cate, cate[obj_candidate])
                            #     if obj_candidate in parentness.keys():
                            #         if int(obj_name.split('/')[-1].split('.')[0]) <  int(parentness[obj_candidate].split('/')[-1].split('.')[0]):
                            #             parentness[parentness[obj_candidate]] = obj_name
                            #         else:
                            #             parentness[obj_name] = parentness[obj_candidate]
                            #             parentness[obj_candidate] = obj_name
                            #     else:
                            #         parentness[obj_candidate] = obj_name

                            #     break
                            feature_dists.append(feature_dist)
                            feature_infos.append(obj_candidate)

                    if len(feature_dists) == 0:
                        continue
                    assert len(feature_dists) == len(feature_infos)
                    min_idx = np.argmin(np.array(feature_dists))
                    # print(feature_dists[min_idx])
                    # raw_input("Enter")
                    #####################dist check######################
                    if feature_dists[min_idx] < 0.13:
                        # if obj_name == './post_box_reid/test_94342_18/124.p' or obj_name =='./post_box_reid/test_94342_18/134.p' \
                        #     or obj_name =='./post_box_reid/test_94342_18/176.p':
                        #     print(obj_name, feature_infos[min_idx])
                        #     raw_input("Enter")
                        obj_candidate = feature_infos[min_idx]
                        print(obj_cate, cate[obj_candidate])

                        if obj_candidate in parentness.keys():
                            if int(obj_name.split('/')[-1].split('.')[0]) < int(
                                    parentness[obj_candidate].split('/')[-1].split('.')[0]):
                                self.switch_key(parentness, obj_candidate, obj_name)
                                # parentness[parentness[obj_candidate]] = obj_name
                            else:
                                parentness[obj_name] = parentness[obj_candidate]
                                parentness[obj_candidate] = obj_name

                        else:
                            parentness[obj_candidate] = obj_name
                        # if obj_name == './post_box_reid/test_94342_18/73.p':
                        #     print(obj_candidate, parentness[obj_candidate])
                        #     print(parentness['./post_box_reid/test_94342_18/134.p'])
                        #     raw_input("Enter")
                with open(save_path + '/' + clip + '.p', 'wb') as f:
                    joblib.dump(parentness, f)

    def merge(self, child, parent):
        child = np.array(child)
        parent = np.array(parent)
        child_avg = np.mean(child, axis=1)
        child[child_avg == 0] = parent[child_avg == 0]
        return child

    def box_neighbor_smooth(self):
        clips = os.listdir(self.save_neighbor_prefix)
        for clip in clips:
            # if not clip == "test_94342_18": # and not clip == "test_94342_7":
            #     continue
            print(clip)
            save_path = self.save_neighbor_smooth_prefix + clip
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue
            with open(os.path.join(self.save_neighbor_prefix, clip) + '/' + clip + '.p', 'rb') as f:
                neighbors = joblib.load(f)
            flags = []
            to_dump = []
            for key, parent in neighbors.items():

                if key in neighbors.values():
                    continue
                link = []
                link.append(key)
                while (parent in neighbors.keys()):
                    link.append(parent)
                    parent = neighbors[parent]
                link.append(parent)
                print(link)
                with open(link[0], 'rb') as f:
                    new_seq = joblib.load(f)
                for obj_id, obj in enumerate(link[:-1]):
                    with open(link[obj_id + 1], 'rb') as f:
                        link_seq = joblib.load(f)
                    new_seq = self.merge(new_seq, link_seq)

                with open(save_path + '/' + link[-1].split('/')[-1], 'wb') as f:
                    joblib.dump(new_seq, f)

                for old_seq in link[:-1]:
                    to_dump.append(old_seq)
            with open(save_path + '/' + 'to_dump.p', 'wb') as f:
                joblib.dump(to_dump, f)

    def box_smooth(self, box_path, save_path_prefix, threshold=10):
        clips = os.listdir(box_path)
        for clip in clips:
            print(clip)
            save_path = save_path_prefix + clip
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue

            obj_names = sorted(glob.glob(os.path.join(box_path, clip) + '/*.p'))
            for obj_name in obj_names:
                print(obj_name)
                with open(obj_name, 'rb') as f:
                    frames = joblib.load(f)
                box_new = []
                temp_id = None
                for frame_id, frame in enumerate(frames):
                    frame = np.array(frame)
                    box_new.append(frame)
                    if np.mean(frame) == 0:
                        if temp_id == None:
                            temp_id = frame_id
                        continue
                    elif not temp_id == None:
                        if temp_id > 3:
                            if frame_id - temp_id < threshold:
                                for j in range(temp_id, frame_id):
                                    if np.linalg.norm(
                                            np.array(box_new[frame_id]) - np.array(box_new[temp_id - 1])) > 30:
                                        box_new[j] = box_new[temp_id - 1]
                                    else:
                                        box_new[j] = (box_new[frame_id] - box_new[temp_id - 1]) / (
                                                    frame_id - temp_id + 1) * (j - temp_id + 1) + box_new[temp_id - 1]
                            # elif frame_id - temp_id < len(frames)/3 and frame_id > 20:
                            #     for j in range(temp_id, frame_id):
                            #         box_new[j] = box_new[temp_id -1]
                        temp_id = None

                with open(save_path + '/' + obj_name.split('/')[-1], 'wb') as f:
                    joblib.dump(box_new, f)

    def visualize_box(self, box_path='./post_neighbor_smooth_newseq/', save_path='./vis_post_neighbor_smooth_newseq/'):
        clips = os.listdir(box_path)
        for clip in clips:
            print(clip)
            obj_names = sorted(glob.glob(os.path.join(box_path, clip) + '/*.p'))
            all_objs = dict()
            for obj_name in obj_names:
                with open(obj_name, 'rb') as f:
                    all_objs[obj_name] = joblib.load(f)

            with open("./to_track/" + clip + '/seqinfo.ini', 'rb') as f:
                infos = f.readlines()
            image_dir = infos[2].split('=')[1][:-1]
            img_names = sorted(glob.glob(image_dir + '/*.jpg'))
            # img_names = sorted(glob.glob(self.img_path + clip + '/img1/*.jpg'))
            if not os.path.exists(save_path + clip):
                os.makedirs(save_path + clip)
            else:
                continue
            for frame_id, img_name in enumerate(img_names):
                img = cv2.imread(img_name)
                save_img_name = img_name.split('/')[-1]
                for obj_name in obj_names:
                    print(frame_id)
                    print(obj_name)
                    obj_box = all_objs[obj_name][frame_id]
                    color = (255, 0, 255)
                    cv2.rectangle(img, (int(obj_box[0]), int(obj_box[1])),
                                  (int(obj_box[0] + obj_box[2]), int(obj_box[1] + obj_box[3])), color, thickness=3)
                    cv2.putText(img, obj_name.split('/')[-1].split('.')[0], (int(obj_box[0]), int(obj_box[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imwrite(save_path + clip + '/' + save_img_name, img)

    def visualize_box_with_neighbor(self, box_path='./post_box_reid/', save_path='./vis_neighbor_box_reid/'):
        clips = os.listdir(box_path)
        for clip in clips:
            # if not clip == "test_94342_18": #not clip == "test_94342_18" and
            #     continue
            print(clip)
            obj_names = sorted(glob.glob(os.path.join(box_path, clip) + '/*.p'))
            all_objs = dict()
            with open(self.save_neighbor_smooth_prefix + clip + '/' + 'to_dump.p', 'rb') as f:
                to_dump = joblib.load(f)
            for obj_name in obj_names:
                if os.path.exists(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1]):
                    with open(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1], 'rb') as f:
                        # print(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1])
                        all_objs[obj_name] = joblib.load(f)
                else:
                    with open(obj_name, 'rb') as f:
                        all_objs[obj_name] = joblib.load(f)

            with open("./to_track/" + clip + '/seqinfo.ini', 'rb') as f:
                infos = f.readlines()
            image_dir = infos[2].split('=')[1][:-1]
            img_names = sorted(glob.glob(image_dir + '/*.jpg'))
            # img_names = sorted(glob.glob(self.img_path + clip + '/img1/*.jpg'))
            if not os.path.exists(save_path + clip):
                os.makedirs(save_path + clip)
            else:
                continue
            for frame_id, img_name in enumerate(img_names):
                img = cv2.imread(img_name)
                save_img_name = img_name.split('/')[-1]
                for obj_name in obj_names:
                    if obj_name in to_dump:
                        continue
                    obj_box = all_objs[obj_name][frame_id]
                    color = (255, 0, 255)
                    cv2.rectangle(img, (int(obj_box[0]), int(obj_box[1])),
                                  (int(obj_box[0] + obj_box[2]), int(obj_box[1] + obj_box[3])), color, thickness=5)
                    cv2.putText(img, obj_name.split('/')[-1].split('.')[0], (int(obj_box[0]), int(obj_box[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imwrite(save_path + clip + '/' + save_img_name, img)

    def generate_newseq(self, box_path='./post_box_reid/', save_path='./neighbor_smooth_newseq/'):
        clips = os.listdir(box_path)
        for clip in clips:
            # if not clip == "test_boelter_3":
            #     continue
            print(clip)
            obj_names = sorted(glob.glob(os.path.join(box_path, clip) + '/*.p'))
            all_objs = dict()
            if not os.path.exists(self.save_neighbor_smooth_prefix + clip + '/' + 'to_dump.p'):
                continue
            with open(self.save_neighbor_smooth_prefix + clip + '/' + 'to_dump.p', 'rb') as f:
                to_dump = joblib.load(f)
            for obj_name in obj_names:
                if os.path.exists(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1]):
                    with open(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1], 'rb') as f:
                        # print(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1])
                        all_objs[obj_name] = joblib.load(f)
                else:
                    print(obj_name)
                    with open(obj_name, 'rb') as f:
                        all_objs[obj_name] = joblib.load(f)
                        # if obj_name == "./post_box_reid/test_boelter_3/12.p":
                        #     print(all_objs[obj_name])

            if not os.path.exists(save_path + clip):
                os.makedirs(save_path + clip)
            # else:
            #     continue
            for obj_name in obj_names:
                if obj_name in to_dump:
                    continue
                # if obj_name == "./post_box_reid/test_boelter_3/12.p":
                #     print(all_objs[obj_name])
                with open(save_path + clip + '/' + obj_name.split('/')[-1], 'wb') as f:
                    joblib.dump(all_objs[obj_name], f)


if __name__ == '__main__':
    box_smoother = BoxSmooth()
    # save each individual objects into separate files
    box_smoother.box_rename()
    # visualize tracking results
    box_smoother.visualize_box(box_path = box_smoother.save_prefix, save_path = './vis_box_reid/')
    # interpolate bounding box for missing frames
    box_smoother.box_smooth(box_smoother.save_prefix, box_smoother.save_smooth_prefix, threshold=10)
    # from box_frame_record() to visualize box, we want to optimize tracking results by merging different tracklets that belong to the same object
    ## find showing and disappearing frames of each object
    box_smoother.box_frame_record()
    ## record the category of each object
    box_smoother.box_category_record()
    ## using category and overlapping of showing and disappearing frames, find tracklets that belong to the same object
    box_smoother.box_find_neighbor()
    box_smoother.box_neighbor_smooth()
    box_smoother.visualize_box_with_neighbor()
    box_smoother.generate_newseq()
    ## final smooth for missing frames and visualize
    box_smoother.box_smooth(box_smoother.neighbor_smooth_newseq, box_smoother.post_smooth_newseq, threshold=30)
    box_smoother.visualize_box()






