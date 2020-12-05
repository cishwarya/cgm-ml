import cv2 
import numpy as np
import pandas as pd
import time
import os,sys,inspect
import matplotlib.pyplot as plt
from PIL import Image

def init(proto, model):
    global net
    print('proto ', proto)
    
    net = cv2.dnn.readNetFromCaffe(proto, model)
    print('cv2 dnn readNetFromCaffe')
    return net

def setPoseDetails(datasetType):
    BODY_PARTS = {}
    POSE_PAIRS = []
    defaultDatasetType = 'default-dataset'

    if datasetType == 'COCO':
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    elif datasetType=='MPI':
        BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                        "Background": 15 }

        POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                        ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                        ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                        ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    else:
        datasetType = defaultDatasetType
        BODY_PARTS ={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,"LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,"LKnee":13,"LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,"LSmallToe":20,"LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background":25}

        POSE_PAIRS =[ ["Neck","MidHip"],   ["Neck","RShoulder"],   ["Neck","LShoulder"],   ["RShoulder","RElbow"],   ["RElbow","RWrist"],   ["LShoulder","LElbow"],   ["LElbow","LWrist"],   ["MidHip","RHip"],   ["RHip","RKnee"],  ["RKnee","RAnkle"], ["MidHip","LHip"],  ["LHip","LKnee"], ["LKnee","LAnkle"],  ["Neck","Nose"],   ["Nose","REye"], ["REye","REar"],  ["Nose","LEye"], ["LEye","LEar"],   
    ["RShoulder","REar"],  ["LShoulder","LEar"],   ["LAnkle","LBigToe"],["LBigToe","LSmallToe"],["LAnkle","LHeel"], ["RAnkle","RBigToe"],["RBigToe","RSmallToe"],["RAnkle","RHeel"] ]

    datasetTypeAndModel = datasetType + '-caffemodel'
    return datasetTypeAndModel, BODY_PARTS, POSE_PAIRS

def addColumnsToDataframe(BODY_PARTS, POSE_PAIRS, df):
    pairCols = []
    
    for i in range(len(POSE_PAIRS)):
        partFrom = POSE_PAIRS[i][0]
        partTo = POSE_PAIRS[i][1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        colname = "P"+str(idFrom)+str(idTo)
        pairCols.append(colname)
                
    #df = df.reindex(columns = df.columns.tolist() + pairCols) 
    df = pd.DataFrame(columns = df.columns.tolist() + pairCols)

    return df, pairCols    

def poseEstimate(imagePath, net, BODY_PARTS, POSE_PAIRS,
    threshold=0.1, width=368, height=368):
    
    points = []
    start_t = time.time()
    
    try:
        #print('imagePath - ', imagePath)
        imgRead = cv2.imread(imagePath)
        frame = cv2.rotate(imgRead, cv2.cv2.ROTATE_90_CLOCKWISE)
        pts = []
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height),
                                    (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        #start_t = time.time()
        out = net.forward()

        print("time is ",time.time()-start_t)

        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            pts.append((int(x), int(y)) if conf > threshold else None)
        
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
            if pts[idFrom] and pts[idTo]:
                #so we keep the points (belonging to pairs together)
                currSetofPoints = [pts[idFrom], pts[idTo]]

                #NOTE: uncomment for DEBUG purposes
                #print('currSetofPoints for ', idFrom, idTo)
                #print('currSetofPoints ', currSetofPoints)

                points.append(currSetofPoints)
                #print('points ', points)
                                
                # the last pairs, are
                # - right ankle to right big toe 
                # - left ankle to left big toe
            else:
                #append [] if no pose points estimated
                points.append([])
                
    except:
        filename = 'outputs/notprocessed.txt'

        if os.path.exists(filename):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # create a new file if not

        expwrite = open(filename,append_write)
        expwrite.write("File: " + imagePath + '\n')
        expwrite.close() 
    return points

    
