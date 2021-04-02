import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result
import mmcv
import glob
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args

def check(obj):
    flag = 0
    obj_ = []
    for i in range(obj.shape[0]):
        score = obj[i, 4]
        if score > 0.5:
            flag = 1
            obj_.append(obj[i])
    return flag, obj_
 

def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=torch.device('cuda', args.device))

    img_path = '../post_images/tracker/'
    save_prefix = '../post_images/tracker_obj/'
    img_folders = os.listdir(img_path)
    for img_folder in img_folders:
        print('**********' + img_path + img_folder + '************')
        clips = glob.glob(img_path + img_folder + '/*')
        for clip in clips:
            print('************' + clip + '****************')
            color_img_names = sorted(glob.glob(clip + '/*.jpg'))
            save_path = save_prefix + clip.split('/')[-1] 
            obj_name = clip.split('/')[-1]
            obj_path = save_prefix + 'objs'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(obj_path):
                os.makedirs(obj_path)
            objects = []
            for idx, color_img_name in enumerate(color_img_names):
                objects.append([])
                print(color_img_name)
                color = cv2.imread(color_img_name)
                result = inference_detector(model, color)
                bbox_result, _ = result
                # bboxes = np.vstack(bbox_result)
                for idt, obj in enumerate(bbox_result):
                    flag, obj_ = check(obj)
                    if (obj.shape is not 0) and flag:
                        print(model.CLASSES[idt])
                        objects[idx].append([model.CLASSES[idt], obj_])

                show_result(
                    color, result, model.CLASSES, score_thr=args.score_thr, wait_time=1, show=False, out_file=save_path+'/'+'{0:04}'.format(idx)+'.jpg')
            with open(obj_path + '/' + obj_name + '.p', 'wb') as f:
                pickle.dump(objects, f, protocol=2)

if __name__ == '__main__':
    main()

#python img_detect.py ../configs/hrnet/htc_hrnetv2p_w32_20e.py ../checkpoints/htc_hrnetv2p_w32_20e_20190810-82f9ef5a.pth
#sudo nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v /home/shuwen/data/data_preprocessing/post_images:/mmdetection/post_images -v /home/shuwen/temp/mmdetection/checkpoints:/mmdetection/checkpoints --ipc=host --network host mmdetection
