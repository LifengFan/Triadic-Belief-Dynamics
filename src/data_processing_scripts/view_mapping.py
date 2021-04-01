import os
import glob
import joblib
import cv2
import numpy as np


class ViewMap:
    def __init__(self):
        self.kinect_img_path = './post_images/kinect/'
        self.skeleton_path = './post_skeletons/'
        self.kinect_box_path = './post_neighbor_smooth_newseq/'
        self.kinect_point_path = './pointclouds/'
        self.tracker_img_path = './post_images/tracker/'
        self.tracker_box_path = './detected_imgs/tracker/objs/'
        self.tracker_mask_path = './detected_imgs/tracker/masks/'
        self.save_neighbor_smooth_prefix = './neihbor_smooth/'
        self.cate_path = './track_cate/'
        self.mask_cate_path = './track_cate_with_frame/'
        self.save_img_path = './view_mapping_record/'
        self.obj_refer = {'handbag': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'bowl': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'suitcase': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'backpack': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'apple': ['apple', 'sports ball'],
                          'sports ball': ['apple', 'sports ball'],
                          'cup': ['remote', 'bottle', 'cup', 'wine glass'],
                          'remote': ['cup', 'bottle', 'remote', 'wine glass'],
                          'bottle': ['cup', 'remote', 'bottle', 'wine glass'],
                          'wine glass': ['cup', 'remote', 'bottle', 'wine glass'],
                          'banana': ['sandwich', 'teddy bear', 'banana'],
                          'sandwich': ['banana', 'teddy bear', 'sandwich'],
                          'teddy bear': ['sandwich', 'banana', 'teddy bear']}
        self.obj_refer_94342 = {'handbag': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'bowl': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'suitcase': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'backpack': ['handbag', 'suitcase', 'backpack', 'bowl'],
                          'apple': ['apple', 'sports ball'],
                          'sports ball': ['apple', 'sports ball'],
                          'cup': ['remote', 'bottle', 'cup', 'wine glass'],
                          'remote': ['cup', 'bottle', 'remote', 'wine glass'],
                          'bottle': ['cup', 'remote', 'bottle', 'wine glass'],
                          'wine glass': ['cup', 'remote', 'bottle', 'wine glass'],
                          'sandwich': ['teddy bear', 'sandwich'],
                          'teddy bear': ['sandwich', 'teddy bear']}
        self.obj_refer_b4 = {'handbag': ['handbag', 'suitcase', 'backpack'],
                             'suitcase': ['handbag', 'suitcase', 'backpack'],
                             'backpack': ['handbag', 'suitcase', 'backpack'],
                             'apple': ['apple', 'sports ball'],
                             'sports ball': ['apple', 'sports ball'],
                             'cup': ['bottle', 'wine glass', 'cell phone', 'book'],
                             'bottle': ['cup', 'wine glass', 'cell phone', 'book'],
                             'wine glass': ['cup', 'bottle', 'cell phone', 'book'],
                             'book': ['tv'],
                             'tv': ['book'],
                             'banana': ['sandwich', 'teddy bear', 'donut'],
                             'sandwich': ['banana', 'teddy bear', 'donut'],
                             'donut': ['banana', 'teddy bear', 'sandwich'],
                             'teddy bear': ['sandwich', 'banana', 'donut']}
        self.obj_refer_b = {'handbag': ['handbag', 'suitcase', 'backpack'],
                             'suitcase': ['handbag', 'suitcase', 'backpack', 'chair'],
                             'backpack': ['handbag', 'suitcase', 'backpack', 'chair'],
                             'apple': ['apple', 'sports ball', 'bowl'],
                             'bowl': ['apple', 'banana'],
                             'book': ['tv'],
                             'sports ball': ['apple', 'sports ball'],
                             'laptop':['book'],
                             'cup': ['bottle', 'wine glass', 'cell phone', 'remote'],
                             'bottle': ['cup', 'wine glass', 'cell phone', 'remote'],
                             'cell phone': ['cup', 'wine glass', 'bottle'],
                             'wine glass': ['cup', 'bottle', 'cell phone'],
                             'banana': ['sandwich', 'teddy bear', 'donut', 'bowl'],
                             'sandwich': ['banana', 'teddy bear', 'donut'],
                             'donut': ['banana', 'teddy bear', 'sandwich', 'dog' , 'person'],
                             'teddy bear': ['sandwich', 'banana', 'donut', 'dog', 'person']}
        self.trackers = {'test1': 'skele1.p', 'test2': 'skele2.p', 'test6': 'skele2.p', 'test7': 'skele1.p',
                         'test_9434_1': 'skele2.p', 'test_9434_3': 'skele2.p', 'test_9434_18': 'skele1.p',
                         'test_94342_0': 'skele2.p', 'test_94342_1': 'skele2.p', 'test_94342_2': 'skele2.p',
                         'test_94342_3': 'skele2.p', 'test_94342_4': 'skele1.p', 'test_94342_5': 'skele1.p',
                         'test_94342_6': 'skele1.p', 'test_94342_7': 'skele1.p', 'test_94342_8': 'skele1.p',
                         'test_94342_10': 'skele2.p', 'test_94342_11': 'skele2.p', 'test_94342_12': 'skele1.p',
                         'test_94342_13': 'skele2.p', 'test_94342_14': 'skele1.p', 'test_94342_15': 'skele2.p',
                         'test_94342_16': 'skele1.p', 'test_94342_17': 'skele2.p', 'test_94342_18': 'skele1.p',
                         'test_94342_19': 'skele2.p', 'test_94342_20': 'skele1.p', 'test_94342_21': 'skele2.p',
                         'test_94342_22': 'skele1.p', 'test_94342_23': 'skele1.p', 'test_94342_24': 'skele1.p',
                         'test_94342_25': 'skele2.p', 'test_94342_26': 'skele1.p',
                         'test_boelter_1': 'skele2.p', 'test_boelter_2': 'skele2.p', 'test_boelter_3': 'skele2.p',
                         'test_boelter_4': 'skele1.p', 'test_boelter_5': 'skele1.p', 'test_boelter_6': 'skele1.p',
                         'test_boelter_7': 'skele1.p', 'test_boelter_9': 'skele1.p', 'test_boelter_10': 'skele1.p',
                         'test_boelter_12': 'skele2.p', 'test_boelter_13': 'skele1.p', 'test_boelter_14': 'skele1.p',
                         'test_boelter_15': 'skele1.p', 'test_boelter_17': 'skele2.p', 'test_boelter_18': 'skele1.p',
                         'test_boelter_19': 'skele2.p', 'test_boelter_21': 'skele1.p', 'test_boelter_22': 'skele2.p',
                         'test_boelter_24': 'skele1.p', 'test_boelter_25': 'skele1.p',
                         'test_boelter2_0': 'skele1.p', 'test_boelter2_2': 'skele1.p', 'test_boelter2_3': 'skele1.p',
                         'test_boelter2_4': 'skele1.p', 'test_boelter2_5': 'skele1.p', 'test_boelter2_6': 'skele1.p',
                         'test_boelter2_7': 'skele2.p', 'test_boelter2_8': 'skele2.p', 'test_boelter2_12': 'skele2.p',
                         'test_boelter2_14': 'skele2.p', 'test_boelter2_15': 'skele2.p', 'test_boelter2_16': 'skele1.p',
                         'test_boelter2_17': 'skele1.p',
                         'test_boelter3_0': 'skele1.p', 'test_boelter3_1': 'skele2.p', 'test_boelter3_2': 'skele2.p',
                         'test_boelter3_3': 'skele2.p', 'test_boelter3_4': 'skele1.p', 'test_boelter3_5': 'skele2.p',
                         'test_boelter3_6': 'skele2.p', 'test_boelter3_7': 'skele1.p', 'test_boelter3_8': 'skele2.p',
                         'test_boelter3_9': 'skele2.p', 'test_boelter3_10': 'skele1.p', 'test_boelter3_11': 'skele2.p',
                         'test_boelter3_12': 'skele2.p', 'test_boelter3_13': 'skele2.p',
                         'test_boelter4_0': 'skele2.p', 'test_boelter4_1': 'skele2.p', 'test_boelter4_2': 'skele2.p',
                         'test_boelter4_3': 'skele2.p', 'test_boelter4_4': 'skele2.p', 'test_boelter4_5': 'skele2.p',
                         'test_boelter4_6': 'skele2.p', 'test_boelter4_7': 'skele2.p', 'test_boelter4_8': 'skele2.p',
                         'test_boelter4_9': 'skele2.p', 'test_boelter4_10': 'skele2.p', 'test_boelter4_11': 'skele2.p',
                         'test_boelter4_12': 'skele2.p', 'test_boelter4_13': 'skele2.p',
                         }
        # self.temp_path = ["test_boelter_3", "test_boelter_24", "test_94342_24", "test_94342_18", "test_94342_7", "test_boelter4_3"]
        self.temp_path = ["test1"]


    def extract_feature(self, cropped_img):
        hists = np.zeros((np.arange(0, 256).shape[0] - 1) * 3)
        hists[:255] = np.histogram(cropped_img[:, :, 0], bins=np.arange(0, 256), density=True)[0]
        hists[255:255 * 2] = np.histogram(cropped_img[:, :, 1], bins=np.arange(0, 256), density=True)[0]
        hists[255 * 2:255 * 3] = np.histogram(cropped_img[:, :, 2], bins=np.arange(0, 256), density=True)[0]
        return hists


    def extract_feature_mask(self, cropped_img, mask):
        hists = np.zeros((np.arange(0, 256).shape[0] - 1) * 3)
        cropped_img[mask] = cropped_img[mask]*0.5 + [50, 0, 0]

        hists[:255] = np.histogram(cropped_img[:, :, 0][mask], bins=np.arange(0, 256), density=True)[0]
        hists[255:255 * 2] = np.histogram(cropped_img[:, :, 1][mask], bins=np.arange(0, 256), density=True)[0]
        hists[255 * 2:255 * 3] = np.histogram(cropped_img[:, :, 2][mask], bins=np.arange(0, 256), density=True)[0]

        return hists


    def extract_gaze(self, skeleton):
        skeleton = np.array(skeleton)
        a = skeleton[21] - skeleton[24]
        b = skeleton[22] - skeleton[24]
        normal = np.cross(a, b)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        normal = normal + np.array([0, 0.8, 0])
        gaze_center = np.vstack([skeleton[21], skeleton[22], skeleton[24]]).mean(axis=0) + np.array([0, 0.2, 0])
        return normal, gaze_center


    def cal_angle(self, normal, obj):
        if np.linalg.norm(obj) > 0:
            obj = obj / np.linalg.norm(obj)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        cosin = obj.dot(normal)
        return cosin


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


    def extract_alpha(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_range = np.max(img_gray) - np.min(img_gray)
        output_range = 255
        alpha = output_range / float(input_range)
        beta = -np.min(img_gray) * alpha
        return alpha, beta


    def hist_match(self, source, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    def calcHist(self, img):
        h_bins = 50
        s_bins = 60
        histSize = [h_bins, s_bins]

        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges  # concat lists
        # Use the 0-th and 1-st channels
        channels = [0, 1]

        hsv_base = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist_base


    def mapping(self):
        video_folders = os.listdir(self.tracker_img_path)
        for video_folder in video_folders:
            clips = os.listdir(self.tracker_img_path + video_folder)
            for clip in clips:
                # if not (len(clip.split('_')) > 1 and (clip.split('_')[1] == "boelter4")):
                #     continue
                # if not clip in self.temp_path:
                #     continue
                if not os.path.exists(os.path.join(self.kinect_box_path, clip)):
                    continue
                if not os.path.exists(self.kinect_point_path + clip):
                    continue
                if not os.path.exists(self.cate_path + clip):
                    continue
                if not os.path.exists(self.tracker_box_path + clip + '.p'):
                    continue
                if not os.path.exists(self.kinect_point_path + clip):
                    continue

                save_path = self.save_img_path + clip
                if not os.path.exists(self.save_img_path + clip):
                    os.makedirs(save_path)
                # else:
                #     continue
                print(clip)
                if len(clip.split('_')) > 1 and clip.split('_')[1] == "boelter4":
                    obj_refer = self.obj_refer_b4
                elif len(clip.split('_')) > 1 and (clip.split('_')[1] == "boelter" or clip.split('_')[1] == "boelter2"
                        or clip.split('_')[1] == "boelter3"):
                    obj_refer = self.obj_refer_b
                elif len(clip.split('_')) > 1 and clip.split('_')[1] == "94342":
                    obj_refer = self.obj_refer_94342
                else:
                    obj_refer = self.obj_refer

                kinect_img_names = sorted(glob.glob(os.path.join(self.kinect_img_path, video_folder, clip) + '/*.jpg'))
                tracker_img_names = sorted(
                    glob.glob(os.path.join(self.tracker_img_path, video_folder, clip) + '/*.jpg'))

                with open(self.tracker_box_path + clip + '.p', 'rb') as f:
                    tracker_boxes = joblib.load(f)

                with open(os.path.join(self.tracker_img_path, video_folder, clip) + '/' + clip + '.p', 'rb') as f:
                    gazes = joblib.load(f)

                obj_names = sorted(glob.glob(self.kinect_box_path + clip + '/*.p'))
                all_objs = dict()
                for obj_name in obj_names:
                    # if os.path.exists(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1]):
                    #     with open(self.save_neighbor_smooth_prefix + clip + '/' + obj_name.split('/')[-1], 'rb') as f:
                    #         all_objs[obj_name] = joblib.load(f)
                    # else:
                    with open(obj_name, 'rb') as f:
                        all_objs[obj_name] = joblib.load(f)

                with open(self.cate_path + clip + '/' + clip + '.p', 'rb') as f:
                    obj_cates = joblib.load(f)

                with open(self.mask_cate_path + clip + '/' + clip + '.p', 'rb') as f:
                    mask_cates = joblib.load(f)

                with open(self.skeleton_path + '/' + clip + '/' + self.trackers[clip], 'rb') as f:
                    skeletons = joblib.load(f)

                with open(self.tracker_mask_path + '/' + clip + '.p', 'rb') as f:
                    tracker_masks = joblib.load(f)

                point_names = sorted(glob.glob(self.kinect_point_path + clip + '/*.p'))

                kinect_target = []
                for frame_id, tracker_box in enumerate(tracker_boxes):
                    frame_id = frame_id
                    skeleton = skeletons[frame_id]
                    if np.mean(np.array(skeleton)) == 0:
                        kinect_target.append(None)
                        continue
                    kinect_img = cv2.imread(kinect_img_names[frame_id])

                    print(tracker_img_names[frame_id])
                    traker_img = cv2.imread(tracker_img_names[frame_id])
                    gaze = np.array(gazes[frame_id]['gaze'])
                    if np.isnan(gaze[0]) or np.isnan(gaze[1]):
                        kinect_target.append(None)
                        continue

                    candidates = []
                    later_use = []
                    for obj_id, obj in enumerate(tracker_box):
                        if obj[0] == 'dining table' or obj[0] == 'couch' or obj[0] == 'chair' or obj[0] == 'bench':
                            for sub_id, sub_obj in enumerate(obj[1]):
                                if gaze[1] <= sub_obj[3] and gaze[1] >= sub_obj[1] and gaze[0] <= sub_obj[2] and gaze[
                                    0] >= sub_obj[0]:
                                    later_use.append([obj[0], sub_obj, obj_id, sub_id])
                            continue
                        for sub_id, sub_obj in enumerate(obj[1]):
                            if gaze[1] <= sub_obj[3] and gaze[1] >= sub_obj[1] and gaze[0] <= sub_obj[2] and gaze[0] >= \
                                    sub_obj[0]:
                                candidates.append([obj[0], sub_obj, obj_id, sub_id])

                    if len(candidates) == 0:
                        for later_cate, later_sub, later_obj, later_sub_id in later_use:
                            if gaze[1] <= later_sub[3] and gaze[1] >= later_sub[1] and gaze[0] <= later_sub[2] and gaze[
                                0] >= later_sub[0]:
                                candidates.append([later_cate, later_sub, later_obj, later_sub_id])

                    if len(candidates) == 0:
                        kinect_target.append(None)
                        continue
                    elif len(candidates) == 1:
                        gaze_box = candidates[0][1]
                        gaze_cate = candidates[0][0]
                        gaze_info = candidates[0][2:]
                    else:
                        min_dist = 100
                        min_id = None
                        min_info = None
                        for sub_id, [cate, sub_obj, obj_id, sub_obj_id] in enumerate(candidates):
                            center = np.array([int((sub_obj[0] + sub_obj[2]) / 2), int((sub_obj[1] + sub_obj[3]) / 2)])
                            perimeter = (sub_obj[2] - sub_obj[0] + sub_obj[3] - sub_obj[1]) * 2
                            dist = np.linalg.norm(center - gaze) / perimeter
                            to_draw = traker_img.copy()

                            if min_dist > dist:
                                min_dist = dist
                                min_id = sub_id
                                min_info = [obj_id, sub_obj_id]

                        gaze_box = candidates[min_id][1]
                        gaze_cate = candidates[min_id][0]
                        gaze_info = min_info
                    # print(gaze_cate)
                    # to_draw = traker_img.copy()
                    # cv2.rectangle(to_draw, (int(gaze_box[0]), int(gaze_box[1])), (int(gaze_box[2]), int(gaze_box[3])), (255, 0, 0), thickness=5)
                    # cv2.circle(to_draw, (int(gaze[0]), int(gaze[1])), 7, (0, 255, 0), thickness=5)
                    # cv2.imshow('img', to_draw)
                    # cv2.waitKey(20)
                    # raw_input('Enter')

                    gaze_box[gaze_box < 0] = 0
                    tracker_cropped_img = traker_img[int(gaze_box[1]):int(gaze_box[3]),
                                          int(gaze_box[0]):int(gaze_box[2]), :]

                    kinect_img_tras = kinect_img.copy()

                    # 1
                    # for i in range(3):
                    #     kinect_img_tras[:,:,i] = self.hist_match(kinect_img[:,:,i], traker_img[:,:,i])
                    # kinect_img_tras[kinect_img_tras>255] = 255

                    # 2
                    # alpha_t, beta_t = self.extract_alpha(traker_img)

                    # alpha_k, beta_k = self.extract_alpha(kinect_img)

                    # kinect_img_tras = np.uint8((kinect_img - beta_k)/alpha_k*alpha_t + beta_t)

                    # alpha_tras, beta_tras = self.extract_alpha(kinect_img_tras)
                    # print(alpha_t, beta_t)
                    # print(alpha_tras, beta_tras)

                    # 3
                    # kinect_img_tras -= np.uint8(np.mean(kinect_img_tras) - np.mean(traker_img))
                    # kinect_img_tras[kinect_img_tras>255] = 255

                    tracker_mask = tracker_masks[frame_id][gaze_info[0]][1][gaze_info[1]]
                    # print(tracker_mask)
                    # if len(tracker_mask[tracker_mask>0]) > 0:
                    #     gaze_hists = self.extract_feature_mask(traker_img, tracker_mask.todense())
                    # else:
                    #     gaze_hists = self.extract_feature(tracker_cropped_img)

                    # gaze_hists = self.extract_feature(tracker_cropped_img)
                    gaze_hists = self.calcHist(tracker_cropped_img)

                    skeleton = skeletons[frame_id]
                    gaze_normal, gaze_center = self.extract_gaze(skeleton)
                    with open(point_names[frame_id], 'rb') as f:
                        mask_objs = joblib.load(f)
                    kinect_features = []
                    for obj_name in obj_names:

                        box = all_objs[obj_name][frame_id]
                        if np.mean(np.array(box)) == 0:
                            continue
                        key = './post_box_reid/' + clip + '/' + obj_name.split('/')[-1]
                        if not (obj_cates[key][0] == gaze_cate or (
                                gaze_cate in obj_refer.keys() and obj_cates[key][0] in obj_refer[gaze_cate])):
                            continue

                        if mask_cates[key][frame_id][0] == None:
                            continue

                        mask = mask_objs[mask_cates[key][frame_id][1]][1][mask_cates[key][frame_id][2]]
                        if key == './post_box_reid/test_9434_1/4.p':
                            print(len(mask[mask>0]))
                        if np.mean(np.array(mask)) == 0:
                            continue

                        avg_col = np.array(mask).mean(axis=1)
                        obj_center = np.array(mask)[avg_col != 0, :].mean(axis=0)
                        gaze_angle = self.cal_angle(gaze_normal, obj_center - gaze_center)
                        # center_screen = self.point2screen(gaze_center)
                        # gaze_screen = self.point2screen(gaze_normal + gaze_center)
                        # obj_screen = self.point2screen(obj_center)
                        # to_draw = kinect_img.copy()
                        # cv2.line(to_draw, (int(gaze_screen[0]), int(gaze_screen[1])), (int(center_screen[0]), int(center_screen[1])), (255, 0, 0), thickness=3)
                        # cv2.line(to_draw, (int(obj_screen[0]), int(obj_screen[1])), (int(center_screen[0]), int(center_screen[1])), (255, 0, 255), thickness=3)
                        # cv2.imshow('img', to_draw)
                        # cv2.waitKey(20)
                        # print(gaze_angle)
                        # raw_input("Enter")
                        # if key == './post_box_reid/test_9434_1/4.p':
                        #     print(gaze_angle)
                        #     exit(0)
                        if gaze_angle < 0.5:
                            continue

                        box[box < 0] = 0
                        cropped_img = kinect_img_tras[int(box[1]):int(box[1] + box[3]),
                                      int(box[0]):int(box[0] + box[2]), :].copy()
                        print(cropped_img.shape)
                        print(tracker_cropped_img.shape)
                        # for i in range(3):
                        #     cropped_img[:, :, i] = self.hist_match(cropped_img[:, :, i], tracker_cropped_img[:, :, i])
                        cropped_img[cropped_img > 255] = 255
                        cv2.imshow('tracker', tracker_cropped_img)
                        cv2.imshow('kinect', cropped_img)
                        cv2.waitKey(20)
                        print(key, obj_cates[key], gaze_cate)
                        # hists = self.extract_feature(cropped_img)
                        # print(np.linalg.norm(gaze_hists - hists))
                        # if frame_id>466:
                        #     raw_input("Enter")
                        hists = self.calcHist(cropped_img)
                        kinect_features.append([obj_name, hists])

                    if len(kinect_features) == 0:
                        kinect_target.append(None)
                        continue
                    dists = []

                    # print(gaze_cate, obj_cates[key][0])
                    # raw_input("Enter")
                    #
                    for feature in kinect_features:
                        dists.append(cv2.compareHist(gaze_hists, feature[1], 0))
                        # dists.append(np.linalg.norm(gaze_hists - feature[1]))
                    idx = np.argmax(np.array(dists))
                    #######check dist##########
                    print(dists)
                    to_draw = kinect_img.copy()

                    center_screen = self.point2screen(gaze_center)
                    gaze_screen = self.point2screen(gaze_normal + gaze_center)
                    cv2.line(to_draw, (int(gaze_screen[0]), int(gaze_screen[1])),
                             (int(center_screen[0]), int(center_screen[1])), (255, 0, 255), thickness=3)
                    traker_img[tracker_mask.todense()] = traker_img[tracker_mask.todense()] * 0.5 + [50, 0, 0]
                    # if comparison_distribution[idx] > 0.75:
                    if dists[idx] > 0.02:
                        kinect_target.append(kinect_features[idx][0].split('/')[-1].split('.')[0])
                        kinect_box = all_objs[kinect_features[idx][0]][frame_id]
                        key = './post_box_reid/' + clip + '/' + kinect_features[idx][0].split('/')[-1]
                        # if mask_cates[key][frame_id][1] == None:
                        #     continue
                        mask = mask_objs[mask_cates[key][frame_id][1]][1][mask_cates[key][frame_id][2]]
                        # if np.mean(np.array(mask)) == 0:
                        #     continue
                        avg_col = np.array(mask).mean(axis=1)
                        obj_center = np.array(mask)[avg_col != 0, :].mean(axis=0)
                        obj_screen = self.point2screen(obj_center)
                        cv2.line(to_draw, (int(obj_screen[0]), int(obj_screen[1])),
                                 (int(center_screen[0]), int(center_screen[1])), (255, 0, 0), thickness=3)
                        cv2.rectangle(to_draw, (int(kinect_box[0]), int(kinect_box[1])),
                                      (int(kinect_box[0] + kinect_box[2]), int(kinect_box[1] + kinect_box[3])),
                                      (255, 0, 0), thickness=3)
                        cv2.putText(to_draw,
                                    kinect_features[idx][0].split('/')[-1] + ':' +
                                    obj_cates[key][0] + ':' + "{:.2f}".format(dists[idx]), (int(kinect_box[0]), int(kinect_box[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                    else:
                        kinect_target.append(None)

                    # cv2.imshow('tracker', traker_img)
                    # cv2.imshow('img', to_draw)
                    # print(dists[idx])
                    concate_img = np.zeros((traker_img.shape[0], tracker_cropped_img.shape[1] + to_draw.shape[1], 3))
                    concate_img[:tracker_cropped_img.shape[0], :tracker_cropped_img.shape[1], :] = tracker_cropped_img
                    cv2.putText(concate_img, gaze_cate, (0, tracker_cropped_img.shape[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                    concate_img[:to_draw.shape[0], tracker_cropped_img.shape[1]:, :] = to_draw
                    concate_img = np.uint8(concate_img)
                    cv2.imwrite(save_path + '/' + '{0:04}'.format(frame_id) + '.jpg', concate_img)
                    # cv2.imshow('img', concate_img)
                    # cv2.waitKey(20)
                    # raw_input("Enter")
                with open(save_path + '/' + 'target.p', 'wb') as f:
                    joblib.dump(kinect_target, f)

    def gaze_data_reformat(self):
        save_prefix = './gaze_training/'

        clips = os.listdir(self.skeleton_path)
        for clip in clips:
            save_path = save_prefix + clip
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(clip)
            with open(self.skeleton_path + clip + '/' + self.trackers[clip], 'rb') as f:
                skeleton_frames = joblib.load(f)
            with open(self.save_img_path + clip + '/' + 'target.p', 'rb') as f:
                target_frames = joblib.load(f)

            with open(self.mask_cate_path + clip + '/' + clip + '.p', 'rb') as f:
                mask_cates = joblib.load(f)

            point_names = sorted(glob.glob(self.kinect_point_path + clip + '/*.p'))
            kinect_img_names = sorted(glob.glob(os.path.join('../../data_preprocessing2/annotations/', clip) + '/kinect/*.jpg'))
            outputs = []
            target_reformats = []
            for frame_id, skeleton_frame in enumerate(skeleton_frames):
                gaze_normal, gaze_center = self.extract_gaze(skeleton_frame)
                target = target_frames[frame_id]
                if target == None:
                    outputs.append(None)
                    target_reformats.append(None)
                    continue
                with open(point_names[frame_id], 'rb') as f:
                    mask_objs = joblib.load(f)

                key = './post_box_reid/' + clip + '/' + target + '.p'
                mask = mask_objs[mask_cates[key][frame_id][1]][1][mask_cates[key][frame_id][2]]
                avg_col = np.array(mask).mean(axis=1)
                obj_center = np.array(mask)[avg_col != 0, :].mean(axis=0)
                obj_screen = self.point2screen(obj_center)
                center_screen = self.point2screen(gaze_center)
                outputs.append(obj_center - gaze_center)
                target_reformats.append([target, obj_center, gaze_center])
                to_draw = cv2.imread(kinect_img_names[frame_id])
                cv2.line(to_draw, (int(obj_screen[0]), int(obj_screen[1])),
                         (int(center_screen[0]), int(center_screen[1])), (255, 0, 0), thickness=3)
                cv2.imshow("img", to_draw)
                cv2.waitKey(20)
                # raw_input('Enter')

            # assert len(skeleton_frames) == len(outputs)
            # assert len(outputs) == len(target_reformats)
            # with open(save_path + '/input.p', 'wb') as f:
            #     joblib.dump(skeleton_frames, f)
            # with open(save_path + '/output.p', 'wb') as f:
            #     joblib.dump(outputs, f)
            # with open(save_path + '/others.p', 'wb') as f:
            #     joblib.dump(target_reformats, f)




if __name__ == "__main__":
    view_mapper = ViewMap()
    view_mapper.mapping()
    # view_mapper.gaze_data_reformat()