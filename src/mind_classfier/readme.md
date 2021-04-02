
Mind classifier
----

The folder contains scripts for mind classifier.

### training
1. Run `main()` in `annotation_clean_saved_att.py` to get data for different baselines.
2. Event prior is calculated using `distribution_cal.py`. 
3. Mind prior is calculated using `get_mind_distribution.py`.
4. Run `python mind_training_combined.py` for training.

### testing
1. Pretrained model can be found [here](xxxx) and put it in ./cptk
2. Searched event tree can be found [here](xxxx) and put it in ./BestTree/
3. Run `test_raw_data()` in `get_mind_posteriror_ours_saved_att.py` for inference on detected objects.
4. Run `get_gt_data()` in `get_mind_posteriror_ours_saved_att.py` for inference on annotated objects.
5. Run `result_eval()` and `save_roc_plot()` in `get_mind_posterior_ours_saved_att.py` to visualize the results. 




