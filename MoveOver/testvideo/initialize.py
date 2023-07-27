### Importing packages
import sys
# Adds parent directory, MoveOver, to the path 
sys.path.insert(0,'..')

import cv2
import pathlib 
import pickle
import config
import numpy as np
from moveoverlib.functions import *
import matplotlib.pyplot as plt

# TODO:  use cv2 to open window of extracted image frames
#       prompt selction of landmarks (camera view, get input; sky view, repeat sequence)
#       display and store coordinates for json file
# make filenames of skyview and parameters specific to video name

# Create /data and /images folder within ./MoveOver/example
pathlib.Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True) 
pathlib.Path('./images/').mkdir(parents=True, exist_ok=True)

# Provided example video
VID_NAME = '3-01'
VIDEO_FILE = f'../../clips/{VID_NAME}.mp4'

# Camera view and bird view images
CAM_VIEW = f'./images/cam_view{VID_NAME}.jpg'
BIRD_VIEW = './images/I-435_SB.PNG'

# Write binary version of video to new ./data folder
# pickle.dump(VIDEO_FILE, open(f'{config.DATA_PATH}/videopath.p', 'wb'))

# Extract frames from video and save in new ./images folder
extractFrame(VIDEO_FILE, frameno = 0, dest_file = CAM_VIEW)

# TODO: use plt.ginput()

# SRC = np.float32([
#     [581, 727], # Left speed limit
#     [1458, 717], # Right speed limit
#     [800, 430], # Left railing
#     [1578, 411], # Right railing
#     [643, 474], # Sign
# ])

# DST = np.float32([
#     [206, 29], # Left speed limit
#     [41, 75], # Right speed limit
#     [567, 925], # Left railing
#     [287, 1170], # Right railing
#     [543, 724], # Sign
# ])


# print ('  "cameraPoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]],'.format(
#     int(SRC[0][0]), int(SRC[0][1]),
#     int(SRC[1][0]), int(SRC[1][1]),
#     int(SRC[2][0]), int(SRC[2][1]),
#     int(SRC[3][0]), int(SRC[3][1])
# ))

# print ('  "birdEyePoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]],'.format(
#     int(DST[0][0]), int(DST[0][1]),
#     int(DST[1][0]), int(DST[1][1]),
#     int(DST[2][0]), int(DST[2][1]),
#     int(DST[3][0]), int(DST[3][1]),
# ))

# # Creates a (maxtrix) mapping from camera view to bird's eye view
# # and its inverse
# M = cv2.getPerspectiveTransform(SRC[:4], DST[:4])
# Minv = cv2.getPerspectiveTransform(DST[:4], SRC[:4])

# # Show points on original camera and transformed bird's eye
# #! sky view is already provided, how was it made? Grab from google maps?
# BIRD_VIEW = './images/SkyView.jpg'
# _ = displayPoints(SRC, M, CAM_VIEW, BIRD_VIEW)

# # Show transformation from bird to camera
# _ = displayPoints(DST, Minv, BIRD_VIEW,  CAM_VIEW)

# #! What are these; from google maps?
# #! from params.json...
# latlon1 = 45.743893, -122.660828    # "latLonCoordinates"[0] flipped (x0, x1)
# xy1 = 96, 30                        # "birdEyeCoordinates"[0]
# latlon2 = 45.742497, -122.659864    # "latLonCoordinates"[1] flipped (x0, x1)
# xy2 = 635, 1151                     # "birdEyeCoordinates"[1]

# #! Is this also done ahead of time manually?
# MASK_PATH = './images/mask.png'

# # import cav
# sys.path.insert(0,'../..')
# from cav.parameters import Parameters

# #! params.json contains same mystery points from SRC and DST
# params = Parameters()
# params.generateParameters('./params.json')

# # loads mask, scales pixel values to 0-255 range, converts to int
# # TODO: method for automatically or assisting masking
# mask = (255*plt.imread(MASK_PATH)).astype(int)

# #! No blue? example mask is len() == 2 anyway
# #! converts 3d mask to 2d? I think it just helps convert to grayscale
# if (len(mask.shape) == 3) and (mask.shape[2] > 1):
#     mask = mask[:, :, 0]

# plt.imshow(mask, cmap='gray')
# #! Why use unique?
# unique = np.unique(mask, return_counts=True)
# print(unique)

# #! Premade test? Don't understand how lane_mask and unique are related
# #! How was lane_mask found?
# #! Why do the colors used in the mask confirm the parameters?
# if [0] + params.lanes_mask == sorted(unique[0]):
#     print ('OK! Mask parameers defined correctly in params.json.')
# else:
#     print (f'json file : {[0] + params.lanes_mask}')
#     print (f'from image : {sorted(unique)}')
    
    
# #! Writes json file, but uses the json file in cells above?
# #! Seems like frame_view1 and SkyView aren't modified and are given ahead of time
# img = cv2.imread(CAM_VIEW)
# skyview = cv2.imread(BIRD_VIEW)

# jsonfile = '''{{
#   "videoShape" : [{}, {}],
#   "birdEyeViewShape" : [{}, {}],

#   "cameraPoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]],
#   "birdEyePoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]],

#   "birdEyeCoordinates" : [[{}, {}], [{}, {}]],
#   "latLonCoordinates" : [[{}, {}], [{}, {}]],
#   "elevation" : 40,
    
#   "lanes_mask" : {}
# }}'''.format(
#     img.shape[1], img.shape[0], # videoShape
#     skyview.shape[1], skyview.shape[0], # birdEyeViewShape
    
#     int(SRC[0][0]), int(SRC[0][1]), # cameraPoints
#     int(SRC[1][0]), int(SRC[1][1]),
#     int(SRC[2][0]), int(SRC[2][1]),
#     int(SRC[3][0]), int(SRC[3][1]), # cameraPointsEnd 
    
#     int(DST[0][0]), int(DST[0][1]), # birdEyePoints
#     int(DST[1][0]), int(DST[1][1]),
#     int(DST[2][0]), int(DST[2][1]),
#     int(DST[3][0]), int(DST[3][1]), # birdEyePointsEnd
    
#     xy1[0], xy1[1], xy2[0], xy2[1], #birdEyeCoordinates

#     latlon1[1], latlon1[0], #latLonCoordinates
#     latlon2[1], latlon2[0],
    
#     str(sorted(unique[0])[1:]), # lanes_mask
# )

# with open('params.json', 'w') as f:
#     for line in jsonfile.split('\n'):
#         print (line)
#         f.write(line + '\n')

