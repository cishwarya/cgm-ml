import sys
sys.path.append('.')
sys.path.append('../../common/depthmap_toolkit/')
import pcd2depth
import dbutils
import pandas as pd
# import utils
import numpy as np
from glob2 import glob
from azureml.core import Workspace, Dataset
import yaml
import random
import logging
import pickle
import os
import multiprocessing
import pathlib 
from data_utils import CollectQrcodes,lenovo_pcd2depth
from pyntcloud import PyntCloud
import shutil
import utils



## Load the yaml file
with open("parameters.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)


# Parse all the configuration variables

db_file = cfg["database"]['db_connection_file']
number_of_scans =cfg['scans']['scan_amount']
calibration_file = cfg['calibration']['calibration_file']
scangroup= cfg['data']['scangroup']
target_folder = cfg['paths']['target_path']
source = cfg['paths']['source_path']

pcd_path = target_folder+'pointclouds'
if not os.path.exists(pcd_path):
    os.makedirs(pcd_path)
    
depthmap_path = target_folder+'depthmaps'
if not os.path.exists(depthmap_path):
    os.makedirs(depthmap_path)

rgb_path = target_folder+'rgb'
if not os.path.exists(rgb_path):
    os.makedirs(rgb_path)

dataset = CollectQrcodes(db_file)
logging.info("Starting the dataset Preparation:")
data = dataset.get_all_data()
scangroup_data = dataset.get_scangroup_data(data=data,scangroup=scangroup)
scangroup_qrcodes = dataset.get_unique_qrcode(scangroup_data)
new_scangroup_data = dataset.get_usable_data(dataframe=scangroup_qrcodes,amount=number_of_scans,scan_group=scangroup)

#new_scangroup_data = new_scangroup_data[new_scangroup_data['tag'] != 'delete']  #TODO: check for delete tag presence and remove those rows

full_dataset = dataset.merge_qrcode_dataset(new_scangroup_data,scangroup_data)

logging.info("Saving the csv file for EDA notebook.")
full_dataset.to_csv('evaluation.csv',index=False)

# Create the RGB csv file for posenet.

get_posenet_data = dataset.get_posenet_results()
get_rgb_artifacts = dataset.get_artifacts()

get_rgb_qrcodedata = dataset.merge_data_artifacts(full_dataset,get_rgb_artifacts)
get_posenet_results = dataset.merge_data_posenet(full_dataset,get_posenet_data)

get_posenet_results.to_csv("RGB_poseresults.csv",index=False)

#Read the Calibration file and set the required shape fro height and width
calibration = utils.parseCalibration(calibration_file)
Width = utils.setWidth(int(240 * 0.75))
Height = utils.setHeight(int(180 * 0.75))


logging.warning("Processing the data:")

def process_data(rows):
    source_path = source + rows['storage_path']
    qrcode = rows['qrcode']
    # qrcode = rows['storage_path'].split('/')[1]
    pcdfile = rows['storage_path'].split('/')[-1]
    depthmaps = pcd2depth.process(calibration,source_path)
    max_value = depthmaps.max()
    if max_value > 10:
       logging.warning(pcdfile)
       return
    scantype = pcdfile.split('_')[3]
    pickle_file = pcdfile.replace('.pcd','.p')
    labels = np.array([float(rows['height']),float(rows['weight']),float(rows['muac']),rows['age'],rows['sex'],rows['tag'],rows['scan_group']])
    depthmap_target_path  = os.path.join(depthmap_path,qrcode)
    depthmap_complete_path = os.path.join(depthmap_target_path,scantype)
    pathlib.Path(depthmap_complete_path).mkdir(parents=True, exist_ok=True)
    data = (depthmaps,labels)
    depthmap_save_path = depthmap_complete_path+'/'+ pickle_file
    pickle.dump(data, open(depthmap_save_path, "wb"))
    pcd_target_path  = os.path.join(pcd_path,qrcode)
    pcd_complete_path = os.path.join(pcd_target_path,scantype)
    pathlib.Path(pcd_complete_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path,pcd_complete_path)
    return


def process_RGB(rows):
    source_path = source + rows['storage_path']
    qrcode = rows['qrcode']
    # qrcode = rows['storage_path'].split('/')[1]
    imagefile = rows['storage_path'].split('/')[-1]
    scantype = imagefile.split('_')[3]    
    rgb_target_path  = os.path.join(rgb_path,qrcode)
    rgb_complete_path = os.path.join(rgb_target_path,scantype)
    pathlib.Path(rgb_complete_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path,rgb_complete_path)
    return
    
proc = multiprocessing.Pool()

for index, row in full_dataset.iterrows():
    # process_data(row)
    proc.apply_async(process_data, [row]) 
    


proc.close()
proc.join() # Wait for all child processes to close.

proc = multiprocessing.Pool()

for index, row in full_dataset.iterrows():
    # process_RGB(row)
    proc.apply_async(process_RGB, [row]) 
    

proc.close()
proc.join()


