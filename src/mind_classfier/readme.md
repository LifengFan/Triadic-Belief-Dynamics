
Mind classifier
----

The folder contains scripts for mind classifier.

### training
1. Details can be found in main() in annotation_clean_saved_att.py to get data for different baselines
2. Event prior is calculated using distribution_cal.py. Mind prior is calculated using get_mind_distribution.py
3. Run `python mind_training_combined.py` for training

### testing
1. Pretrained model can be found in [here](xxxx) and put it in ./cptk
2. Searched event tree can be found in [here](xxxx) and put it in ./BestTree/
3. inference on detected objects can be run using test_raw_data() in get_mind_posteriror_ours_saved_att.py
4. inference on annotated objects can be run using get_gt_data() in get_mind_posteriror_ours_saved_att.py
5. results can be visualized using result_eval() and save_roc_plot() in get_mind_posterior_ours_saved_att.py




