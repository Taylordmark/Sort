import sys
sys.path.append('/remote_home/Thesis')

import xml.etree.ElementTree as ET
import os

import sort
from pathlib import Path

import pickle

import csv
import math
import numpy as np

folder_path = r"C:\Users\nloftus\Documents\Datasets\RoboflowDataset\RFFiltered2"
output_path = r"gt_3.pkl"

img_size = 512

def bb_intersection_over_union(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou

def match_boxes(boxes, tracks):
    assignments = []
    for track in tracks:
        track_ious = []
        for box in boxes:
            boxa = track[0:4]
            boxb = box
            track_ious.append(bb_intersection_over_union(boxa, boxb))
        track_association = np.argmax(track_ious)
        assignments.append(track_association)

    return assignments

"""
Our model classes are
0: car
1: pedestrian
2: biker
3: truck
4: traffic-light
5: tl-red
6: tl-green
"""


#convert gt classes to our model classes
cls_dict ={
    0: 0, 
    1: 0, #1 to car
    2: 1, #2 to pedestrian 
    3: 6, #3 and 4 to green light
    4: 6,
    5: 5, #5 and 6 to red light
    6: 5,
    7: 4, #7 and 8 to generic light
    8: 4,
    9: 2, #9 to biker
    10: 3 #10 to truck
}

mot_tracker = sort.Sort(min_hits=1, iou_threshold=.4, max_age=2)

#mot_tracker.update()


half = False
skip = True

with open(output_path, 'wb') as pickle_file:
    results_dict = {}
    frame_num = 0


    for filename in os.listdir(folder_path):

        if ".txt" not in filename:
            continue

        if half:
            skip = not skip
            if skip:
                continue

        dets = []
        clses = []



        with open(folder_path+f"\\{filename}", 'r') as file:
            for line in file:
                arr = line.split(" ")
                box = arr[1:5]
                box = [float(box[0])*img_size, float(box[1])*img_size, (float(box[0])+float(box[2]))*img_size, (float(box[1])+float(box[3]))*img_size]
                
                
                dets.append(box)

                if frame_num == 4:
                    print(arr)
                    print(cls_dict[int(arr[0])])
                clses.append(cls_dict[int(arr[0])])



        if (dets != []):
            track_bbs_ids = mot_tracker.update(dets)
        else:
            track_bbs_ids = mot_tracker.update(np.empty((0,5)))


        track_boxes = [[box[0]-(box[2]-box[0])*0.5, box[1]+(box[1]-box[3])*0.5, box[2]-(box[2]-box[0])*0.5, box[3]+(box[1]-box[3])*0.5] for box in track_bbs_ids[:,0:4]]

        assignments = match_boxes(dets, track_boxes)
        assigned_probs = [clses[a] for a in assignments]

        results_dict[frame_num] = {'boxes': track_boxes, 'cls_prob': assigned_probs, 'track_id':track_bbs_ids[:,-1]}


        frame_num += 1
    print(results_dict)

    pickle.dump(results_dict, pickle_file)
    