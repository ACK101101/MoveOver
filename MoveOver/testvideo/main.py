import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'..')

import pyautogui
pyautogui.FAILSAFE = False

import pickle
import config
import pathlib
import os
import cv2
from pathlib import Path
from time import time
from cav.detection import ObjectDetector

### ISSUE: LOOK OUT FOR WHERE params.json IS USED!
pathlib.Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True) 
pathlib.Path('./images/').mkdir(parents=True, exist_ok=True)
VIDEO_NAME = '3-01'
VIDEO_FILE = f'../../videos/{VIDEO_NAME}.mp4'                                               # video run 
pickle.dump(VIDEO_FILE, open(f'{config.DATA_PATH}/vid{VIDEO_NAME}.p', 'wb'))                # binarized video

od = ObjectDetector(config.MODEL_PATH)                                                      # model

SAVE_DETECTIONS = os.path.join(config.DATA_PATH, f'detect{VIDEO_NAME}.p')                   # makes path and directories to save detections
Path(os.sep.join(SAVE_DETECTIONS.split(os.sep)[:-1])).mkdir(parents=True, exist_ok=True)


# VIDEO_FILE = pickle.load(open(f'{config.DATA_PATH}/videopath.p', 'rb'))
CUT_UPPER = 0

cap = cv2.VideoCapture(VIDEO_FILE)

save_detections = {}    

i = 0
start_time = time()

while cap.isOpened():
    pyautogui.moveRel(1, 1)
    pyautogui.moveRel(-1, -1)
    pyautogui.press('left')
    # get width, height, and fps of video
    video_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    video_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - CUT_UPPER     
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_timeStamp = i/fps

    # time since beginning of while loop
    elapsed_time = time() - start_time
    sys.stdout.write('{} frames done in {:.1f} seconds ({:.2f} frames/sec)    \r'.format(
        i, elapsed_time, i/elapsed_time))                   
    i += 1
    
    # gets next frame
    _, image = cap.read()
    # if no more frames left
    if image is None: break

    #! are the 3 dimensions not RGB?
    if CUT_UPPER > 0:
        image = image[CUT_UPPER:, :, :]
    
    #! trying to detect move over event? 
    #! what is it drawing bounding boxes around?
    boxes, scores, classes = od.detect(image, timestamp=frame_timeStamp)
    if SAVE_DETECTIONS is not None:
        save_detections[i] = (boxes, scores, classes)
    
cap.release()

if SAVE_DETECTIONS is not None:
    pickle.dump(save_detections, open(SAVE_DETECTIONS,'wb'))
    print (f'\nDetections saved in {SAVE_DETECTIONS}.')


### HOW TO CHECKPOINT?

