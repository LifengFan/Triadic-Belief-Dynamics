# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import sys
import cv2
import numpy as np
from scipy import sparse
import tqdm
import joblib

import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument('--data-path', default='/mnt/ssd/', help='data path')
    parser.add_argument('--camera-cate', default='kinect', help='project root path')
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    camera_cate = args.camera_cate
    data_path = args.data_path
    img_path = data_path + '/annotations/'
    obj_path = data_path + '/detected_imgs/' + camera_cate + '/objs/'
    mask_path = data_path + '/detected_imgs/' + camera_cate + '/masks/'
    video_path = data_path + '/detected_imgs/' + camera_cate + '/videos/'
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    if not os.path.exists(video_path):
        os.makedirs(video_path)

    clips = os.listdir(img_path)
    for clip in clips:
        demo = VisualizationDemo(cfg)
        print(clip)
        save_path = data_path + clip

        img_path = data_path + '/annotations/' + clip + '/' + camera_cate + '/'
        img_names = sorted(glob.glob(img_path + '*.jpg'))
        args.output = save_path
        objs = []
        obj_masks = []
        filename1 = video_path + '/' + clip + '.mp4'
        out = cv2.VideoWriter(filename1, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24, (1280, 720))
        for path_id, path in enumerate(img_names):
            objs.append([])
            obj_masks.append([])
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output, metadata = demo.run_on_image(img)
            labels = metadata.get("thing_classes", None)
            instances = predictions["instances"].to(torch.device("cpu"))
            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes if instances.has("pred_classes") else None
            masks = instances.pred_masks if instances.has("pred_masks") else None
            boxes = boxes.tensor.numpy()
            scores = scores.numpy()
            classes = classes.numpy()
            masks = masks.numpy()
            cates = np.unique(classes)
            for cate in cates:
                box = boxes[classes==cate]
                score = scores[classes==cate]
                score = score.reshape((score.shape[0], 1))
                obj_info = np.hstack([box, score])
                objs[path_id].append([labels[cate], obj_info])

                mask = masks[classes==cate]
                mask_sp = []
                for sub_mask in mask:
                    mask_sp.append(sparse.csr_matrix(sub_mask))
                obj_masks[path_id].append([labels[cate], mask_sp])


            # logger.info(
            #     "{}: detected {} instances in {:.2f}s".format(
            #         path, len(predictions["instances"]), time.time() - start_time
            #     )
            # )
            out.write(visualized_output.get_image()[:, :, ::-1])
            # if args.output:
            #     if os.path.isdir(args.output):
            #         assert os.path.isdir(args.output), args.output
            #         out_filename = os.path.join(args.output, os.path.basename(path))
            #     else:
            #         assert len(args.input) == 1, "Please specify a directory with args.output"
            #         out_filename = args.output
            #     visualized_output.save(out_filename)
            # else:
            #     cv2.imshow("COCO detections", visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(20) == 27:
            #         break  # esc to quit

        out.release()
        with open(obj_path + clip + '.p', 'wb') as f:
            joblib.dump(objs, f, protocol=2)

        with open(mask_path + clip + '.p', 'wb') as f:
            joblib.dump(obj_masks, f, protocol=2)

