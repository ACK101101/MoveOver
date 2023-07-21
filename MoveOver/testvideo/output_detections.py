import sys
sys.path.insert(0,'..')

import pandas as pd
import numpy as np
import pickle 
import cv2
import math
import os

import config

ACTION_FILE = './data/actions.csv' # output from Lane_analysis notebook
LANE_FILE = './data/lanes_detections.csv' # output from detect_lanes notebook
OUTPUT_FILE = os.getcwd().split('/')[-1] # The output file that may be shared outside
OUTPUT_FILE = f'./data/{OUTPUT_FILE}_output.csv'
print ('Output file:', OUTPUT_FILE)


VIDEO_FILE = pickle.load(open(f'{config.DATA_PATH}/videopath.p', 'rb'))

cap = cv2.VideoCapture(VIDEO_FILE) 
fps = cap.get(cv2.CAP_PROP_FPS)
print ('FPS:', fps)

dfraw = pd.read_csv(LANE_FILE, header=None)
dfraw.columns = ['frame', 'lane', 'objectId', 'objectType', 'secMark', 
                 'xLeft', 'xRight', 'yTop', 'yBottom', 'lat', 'lon', 'speed', 'heading', 'elevation'] 
lane = dfraw.groupby('objectId').agg({
    'frame' : [np.min, np.max],  
    'objectType' : [lambda x:x.value_counts().index[0], 'mean'],
}).reset_index()
lane.columns = ['objectId', 'frame_start', 'frame_end', 'objectType', 'otMean']
lane.head()
lane = lane.merge(dfraw[['objectId', 'frame', 'lat', 'lon']], 
                  left_on = ['objectId', 'frame_start'],
                  right_on = ['objectId', 'frame']).drop('frame', axis=1)

lane = lane.merge(dfraw[['objectId', 'frame', 'lat', 'lon']], 
                  left_on = ['objectId', 'frame_end'],
                  right_on = ['objectId', 'frame']).drop('frame', axis=1)
lane = lane.rename(columns = {
    'lat_x' : 'lat_start',
    'lon_x' : 'lon_start',
    'lat_y' : 'lat_end',
    'lon_y' : 'lon_end',    
})

lane['time'] = (lane.frame_end - lane.frame_start) / fps
lane.head()

def computeDistance(row):
    """
    Computes speeds between two points determined by
    (row.lat_start, row.lon_start), (row.lat_end, row.lon_end)
    Arguments:
        row - a structure (pd.Series) with defined the abovementioned
              lat/lon features
    Returns: 
        distance in meters
    """
    
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - row.lat_start) * degrees_to_radians
    phi2 = (90.0 - row.lat_end) * degrees_to_radians

    # theta = longitude
    theta1 = row.lon_start * degrees_to_radians
    theta2 = row.lon_end * degrees_to_radians

    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) +
       math.cos(phi1) * math.cos(phi2))

    cos = max(-1, min(1, cos)) # in case of numerical problems

    ret = 6731000 * math.acos(cos) # mutliplied by earth radius in meters
    return ret

lane['dist'] = lane.apply(computeDistance, axis=1)
lane.head()

lane['speed'] = lane.dist / lane.time * 2.237 # m/s -> MpH
lane = lane[lane.time > 0]
lane.head()

df = pd.read_csv(ACTION_FILE)
df = df[['objectId', 'action', 'slowed', 'can_change']]
print (df.shape)
df = df.merge(lane, on='objectId')
print (df.shape)
df.head()

df = df[[
    'objectId', 'frame_start', 'frame_end', 'objectType', 
    #'speed', 
    'action', 'can_change', 'slowed']]
df.loc[df.can_change == 1, 'can_change'] = True
df.loc[df.can_change == 0, 'can_change'] = False
df.head()

df.to_csv(OUTPUT_FILE, index=False)

df.slowed.value_counts()