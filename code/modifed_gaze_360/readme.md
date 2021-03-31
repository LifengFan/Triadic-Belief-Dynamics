
Modified gaze 360
----

The folder contains scripts for gaze prediction. We use the 3D gaze direction estimation from the eye tracker as the ground truth and fine tune the gaze 360 model. Then the trained model will be used for gaze prediction for the person wearing Pivothead glasses.

###training
1. training data can generated using record_skeleton_for_training() in test_gaze_360.py
2. Run `python run_skele.py` for training

###inference
Use gaze360_estimation() in test_gaze_360.py for gaze prediction for the person wearing Pivothead glasses

###gaze smooth
After gaze prediction, we interpolate for missing frames. Details can be found in main() gaze_smooth.py


