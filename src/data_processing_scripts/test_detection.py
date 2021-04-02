import cv2
import glob
import numpy as np
import joblib

cate = 'kinect'
path_prefix = './resort_detected_imgs/' + cate + '/'
obj_path = path_prefix + 'objs/'
masks_path = path_prefix + 'masks/'
img_path = './annotations/'

clips = sorted(glob.glob(obj_path + '*.p'))

for clip in clips[:1]:
    with open(clip, 'rb') as f:
        obj_frames = joblib.load(f)
    with open(masks_path + clip.split('/')[-1], 'rb') as f:
        masks_frames = joblib.load(f)
    clip_name = clip.split('/')[-1].split('.')[0]
    img_names = sorted(glob.glob(img_path + clip_name + '/' + cate + '/*.jpg'))
    for frame_id, obj_frame in enumerate(obj_frames):
        img = cv2.imread(img_names[frame_id])
        for obj_id, obj in enumerate(obj_frame):
            for sub_id, sub_obj in enumerate(obj[1]):
                cv2.rectangle(img, (int(sub_obj[0]), int(sub_obj[1])), (int(sub_obj[2]), int(sub_obj[3])), (255, 0, 0),
                              thickness=3)
                cv2.putText(img, obj[0], (int(sub_obj[0]), int(sub_obj[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0),
                            3, cv2.LINE_AA)
                mask = masks_frames[frame_id][obj_id][1][sub_id]
                dense_mask = mask.todense()
                img[dense_mask] = 0.5 * img[dense_mask] + [100, 0, 0]
        cv2.imshow(img_names[frame_id], img)
        cv2.waitKey(200)


