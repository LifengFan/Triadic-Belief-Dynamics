
Data processing scripts
----

The folder contains scripts for data processing, including object detection, object tracking, getting pointclouds from depth image, eye-tracker gaze estimation

###object detection
1. Follow instruction on [Detectron 2](https://github.com/facebookresearch/detectron2) to install necessary libraries
2. Run `python detectron2_img_detect_kinect.py --data-path` to extract object bounding box, mask, and result videos

###object tracking
1. Follow instruction on [Deep SORT](https://github.com/nwojke/deep_sort) to install necessary libraries
2. Change the results of object detection to the format required by deep SORT by running Run `python obj_reformat.py` and `python deep_sort_generate_detections.py --model --mot_dir --output_dir` 
3. Run `python deep_sort_tracker.py --data_path --detection_file --output_path` to get object tracking results

###object tracking smooth
Please refer to main() in box_smooth.py for detailed explanation

###pointclouds from depth image
Taking the object masks estimated from detectron2 and the depth image, we will estimation the 3D center of each object by running `python pointclouds.py`

###eye-tracker gaze estimation
Since the eye-tracker has precise gaze estimation on 2D first-view image, we want to estimate the 3D gaze direction by finding the corresponding objects between first-view and third-view based on the colors and the categories of the objects. Then the difference between the 3D center of the objects (estimated from pointclouds) and the center of the huamn head (estimation from skeleton) can be treated as the gaze direction in 3D space. Details can be found in view_mapping.py


