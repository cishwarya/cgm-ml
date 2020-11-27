import os
import shutil
from glob2 import glob
import multiprocessing


file_type = 'realtime_evaluation_updated/depthmaps/*'
target = '/mnt/depthmap/realtime_testset/scans/'

files = glob(file_type)

def copy_files(scans):
    qrcode = scans.split('/')[2]
    target_folder = target+qrcode+'/'
    shutil.copytree(scans,target_folder)

proc = multiprocessing.Pool()

for row in files:
    # copy_files(row)
    proc.apply_async(copy_files, [row])

proc.close()
proc.join()
