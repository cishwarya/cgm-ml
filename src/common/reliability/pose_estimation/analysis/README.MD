
Pre-requisities:

Pretrained CAFFE Pose estimation model is executed on (1/6)th of the samples from the "anon_RGB_training dataset" and 
the result is stored in anonrgbtrain_poseestimation_ps_posepoints.json.
(anonrgbtrain_poseestimation_ps_posepoints.json is compressed and uploaded. Please unzip and use)

The objective of analyseOutput.py script is to analyse anonrgbtrain_poseestimation_ps_posepoints.json in order to find the missing posepoints 
and thereby calculating the accuracy.
 
Following files are required to run analyseOutput.py:

 1. config.py 
 2. posepoints.py 
 3. anonrgbtrain_poseestimation_ps_posepoints.json

- config.py file contains the list of parameter configuration that has been used to train the pose estimation model
- posepoints.py file contains functions to set pose details (e.g. POSE_PAIRS and BODY_PARTS) which has been used in the analyseOutput.py script

Required libraries to run analyseOutput.py:
- pandas

Tests have been carried on different samples of data and results have been verified. 
Please refer to SampleOutput.txt for a sample output.
