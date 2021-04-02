import os
import joblib
import numpy as np
import glob
import cv2


class BoxSmooth:
    def __init__(self):
        self.box_path = '/home/shuwen/temp/deep_sort/result2/'
        self.save_prefix = './box_reid/'
        self.img_path = './to_track/'
        self.mask_path = './resort_detected_imgs/kinect/masks/'
        self.save_smooth_prefix = './post_box_reid/'
        self.save_frame_prefix = './frame_record/'
        self.save_neighbor_prefix = './neihbor_record/'
        self.save_cat_prefix = './track_cate_with_frame/'
        self.save_neighbor_smooth_prefix = './neihbor_smooth/'
        self.neighbor_smooth_newseq = './neighbor_smooth_newseq/'
        self.post_smooth_newseq = './post_neighbor_smooth_newseq/'


    def box_category_record(self):
        clips = os.listdir(self.save_prefix)
        dete_prefix = './resort_detected_imgs/kinect/objs/'

        for clip in clips:
            print(clip)
            save_path = self.save_cat_prefix + clip
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue
            obj_names = sorted(glob.glob(os.path.join(self.save_smooth_prefix, clip) + '/*.p'))
            with open(dete_prefix + clip + '.p', 'rb') as f:
                dete_frames = joblib.load(f)

            with open("./to_track/" + clip + '/seqinfo.ini', 'rb') as f:
                infos = f.readlines()
            image_dir = infos[2].split('=')[1][:-1]
            kinect_img_names = sorted(glob.glob(image_dir + '/*.jpg'))
            # kinect_img_names = sorted(glob.glob('./to_track/' + clip + '/img1/*.jpg'))
            cate = dict()
            for obj_name in obj_names:
                with open(obj_name, 'rb') as f_:
                    frames = joblib.load(f_)
                cate_frames = []
                for frame_id, frame in enumerate(frames):
                    if np.mean(frame) > 0:
                        dist_min = 1000
                        cat_min = None
                        sub_min = None
                        obj_min = None
                        box_min = None

                        dete_frame = dete_frames[frame_id]
                        img = cv2.imread(kinect_img_names[frame_id])
                        for obj_id, obj in enumerate(dete_frame):
                            for sub_id, sub_obj in enumerate(obj[1]):
                                # box_min = sub_obj[:-1]
                                # cv2.rectangle(img, (int(box_min[0]), int(box_min[1])), (int(box_min[2]), int(box_min[3])), (255,0,0), thickness=5)
                                # cv2.imshow(cat_min, img)
                                # cv2.waitKey(20)
                                # raw_input("Enter")
                                dist = np.linalg.norm(
                                    np.array([frame[0], frame[1], frame[0] + frame[2], frame[1] + frame[3]]) - np.array(
                                        sub_obj[:-1]))
                                if dist < dist_min:
                                    dist_min = dist
                                    cat_min = obj[0]
                                    sub_min = sub_id
                                    obj_min = obj_id
                                    box_min = sub_obj[:-1]
                                    # cv2.rectangle(img, (int(frame[0]), int(frame[1])), (int(frame[0] + frame[2]), int(frame[1] + frame[3])), (255,0,255), thickness=5)
                                    # cv2.rectangle(img, (int(box_min[0]), int(box_min[1])), (int(box_min[2]), int(box_min[3])), (255,0,0), thickness=5)
                                    # cv2.imshow(cat_min, img)
                                    # cv2.waitKey(20)
                                    # raw_input("Enter")
                        cate_frames.append([cat_min, obj_min, sub_min])

                    else:
                        cate_frames.append([None, None, None])

                cate[obj_name] = cate_frames
            with open(save_path + '/' + clip + '.p', 'wb') as f:
                joblib.dump(cate, f)


if __name__ == '__main__':
    box_smoother = BoxSmooth()
    # box_smoother.box_rename()
    # box_smoother.visualize_box(box_path = box_smoother.save_prefix, save_path = './vis_box_reid/')
    # box_smoother.box_smooth(box_smoother.save_prefix, box_smoother.save_smooth_prefix, threshold=10)
    # box_smoother.box_frame_record()
    box_smoother.box_category_record()
    # box_smoother.box_find_neighbor()
    # box_smoother.box_neighbor_smooth()
    # box_smoother.visualize_box_with_neighbor()
    # box_smoother.generate_newseq()
    # box_smoother.box_smooth(box_smoother.neighbor_smooth_newseq, box_smoother.post_smooth_newseq, threshold=30)
    # box_smoother.visualize_box()






