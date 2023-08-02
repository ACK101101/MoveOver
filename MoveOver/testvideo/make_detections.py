import pyautogui
pyautogui.FAILSAFE = False

import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'..')

import pickle
import cv2
import config
from pathlib import Path
from time import time
from cav.detection import ObjectDetector

# TODO: redo folder structure

class MakeDetections():
    def __init__(self, Paths, keep_awake=False):
        """runs and saves detections automatically when called

        Args:
            vidname (str): name of video used to find video mp4
            keep_awake (bool): keep machine awake
            process_every (int): processes every kth frame (default 1, processes every frame)
        """
        self.KEEP_AWAKE = keep_awake
        self.PROCESS_EVERY = Paths.process_every
        self.CUT_UPPER = 0
        self.detections = {}
        self.MODEL_PATH = config.MODEL_PATH
        self.model = ObjectDetector(self.MODEL_PATH)
        
        isContinue = self._check_existing(Paths)
        if isContinue:             
            pickle.dump(Paths.VIDEO_FILE, open(Paths.VIDEO_BINARY, 'wb'))                                   # makes path and directories to save detections                                                  # model

            self._run_model(Paths)
    
    def _check_existing(self, Paths):
        if Path(Paths.SAVE_DETECTIONS).exists():
            print("It looks like the detections for this video already exists.")
            isContinue = input("Do you want to redo them? (y/n): ").lower().strip() == 'y'
            if not isContinue:
                print("Skipping detections")
                return False
        return True
    
    def _keep_awake(self):
        """moves mouse and presses key to keep computer from shutting off
            (very optional and specific to the setup of my machine)
        """
        pyautogui.moveRel(1, 1)
        pyautogui.moveRel(-1, -1)
        pyautogui.press('shift')
        return
    
    def _run_model(self, Paths):
        """passes each frame thru model, saves detections
        """
        # get video and setup 
        cap = cv2.VideoCapture(Paths.VIDEO_FILE) 
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
        pickle.dump(self.detections, open(Paths.SAVE_DETECTIONS,'wb'))
        print (f'\nDetections saved in {Paths.SAVE_DETECTIONS}.')
        
        return