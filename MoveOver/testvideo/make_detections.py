import pyautogui
pyautogui.FAILSAFE = False

import pickle
import os
import sys
import cv2
import config
from pathlib import Path
from time import time
from cav.detection import ObjectDetector

# TODO: consolidate os and sys and pathlib so only need one
# TODO: option to skip frames to reduce runtime  

class MakeDetections():
    def __init__(self, vidname, keep_awake=False, process_every=1):
        """runs and saves detections automatically when called

        Args:
            vidname (str): name of video used to find video mp4
            keep_awake (bool): keep machine awake
            process_every (int): processes every kth frame (default 1, processes every frame)
        """
        self.VIDEO_NAME = vidname
        self.VIDEO_FILE = f'../../clips/{self.VIDEO_NAME}.mp4'
        self.DATA_PATH = config.DATA_PATH
        self.MODEL_PATH = config.MODEL_PATH
        self.KEEP_AWAKE = keep_awake
        self.PROCESS_EVERY = process_every
        self.SAVE_DETECTIONS = os.path.join(self.DATA_PATH, 
                                            f'detect{self.VIDEO_NAME}.p')
        self.CUT_UPPER = 0
        self.detections = {}
        self.model = ObjectDetector(self.MODEL_PATH) 
                       
        pickle.dump(self.VIDEO_FILE, 
                    open(f'{self.DATA_PATH}/vid{self.VIDEO_NAME}.p', 
                         'wb'))                                   # makes path and directories to save detections
        Path(os.sep.join(self.SAVE_DETECTIONS.split(os.sep)[:-1])).mkdir(parents=True, exist_ok=True)                                                     # model

        self._run_model()
    
    def _keep_awake(self):
        """moves mouse and presses key to keep computer from shutting off
            (very optional and specific to the setup of my machine)
        """
        pyautogui.moveRel(1, 1)
        pyautogui.moveRel(-1, -1)
        pyautogui.press('shift')
        return
    
    def _run_model(self):
        """passes each frame thru model, saves detections
        """
        # get video and setup 
        cap = cv2.VideoCapture(self.VIDEO_FILE) 
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        start_time = time()  

        while cap.isOpened():
            if frame_num % self.PROCESS_EVERY == 0:
                if self.KEEP_AWAKE: self._keep_awake()

                # tracks progress of process
                frame_timeStamp = frame_num/fps
                elapsed_time = time() - start_time
                sys.stdout.write(
                    '{} frames done in {:.1f} seconds ({:.2f} frames/sec)    \r'.format(
                    frame_num, elapsed_time, frame_num/elapsed_time))                   
                frame_num += 1
                
                _, image = cap.read()       # gets next frame
                if image is None: break     # if no more frames left

                if self.CUT_UPPER > 0:      # crop
                    image = image[self.CUT_UPPER:, :, :]
                
                # runs model, saves detections
                boxes, scores, classes = self.model.detect(image, timestamp=frame_timeStamp)
                self.detections[frame_num] = (boxes, scores, classes)
            else:
                frame_num += 1
                _, image = cap.read()       # gets next frame
                if image is None: break     # if no more frames left
                
        cap.release()                   # when video is over
        
        # save detections
        pickle.dump(self.detections, open(self.SAVE_DETECTIONS,'wb'))
        print (f'\nDetections saved in {self.SAVE_DETECTIONS}.')
        
        return