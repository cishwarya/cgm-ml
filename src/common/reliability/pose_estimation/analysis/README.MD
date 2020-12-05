
Pre-requisities:

Pretrained CAFFE Pose estimation model is run on (1/6)th of the samples from the anon_RGB_training dataset and 
the result is stored in anonrgbtrain_poseestimation_ps_posepoints.json.
(anonrgbtrain_poseestimation_ps_posepoints.json is compressed and uploaded. Please unzip and use)

The analyseOutput.py script is used for analysing this file in order to find the missing posepoints 
and thereby calculating the accuracy.
 
Following files are required to run this script:

 1. config.py 
 2. posepoints.py 
 3. anonrgbtrain_poseestimation_ps_posepoints.json

- config .py file contains the list of parameter configuration that has been used to run the pose estomation model on the data
- posepoints.py file contains functiosn to set pose details (POSE_PAIRS and BODY_PARTS) which has been used in the analyseOutput.py script

Required libraries to run analyseOutput.py:
- pandas

SampleOutput.txt contains sample output from executing analyseOutput.py
