### Importing packages
import sys
# Adds parent directory, MoveOver, to the path 
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import cv2
import os
import config
from pathlib import Path
import time
import numpy as np
from moveoverlib.functions import *
import matplotlib.pyplot as plt

# TODO: write useful variables to config file
#       OR
#       Make experiment class for each varible
# TODO: Save bird view under better name
# TODO: look into cleaner way to handling paths

class Experiment():
    """ generate config file
        setup file and directory names
        prompt user to select reference points to generate map
        stores paths used in experiment
    """
    def __init__(self, video_info, process_every):
        # Provided example video
        self.VIDEO_NAME = video_info["VideoName"]
        self.DATA_PATH = config.DATA_PATH
        self.IMAGE_PATH = './images/'
        
        self.VIDEO_FILE = f'./clips/{self.VIDEO_NAME}.mp4'
        self.CAM_VIEW = f'./images/cam_view{self.VIDEO_NAME}.jpg'         # setup file name
        self.BIRD_VIEW = f'./images/{video_info["BirdView"]}'           # already exists     
        
        self.PARAM_FILE = f'./{self.VIDEO_NAME}/params{self.VIDEO_NAME}.json'
        self.FRAME_FOLDER = os.path.join(self.DATA_PATH, f'frames_raw{self.VIDEO_NAME}/')
        self.VIDEO_BINARY = os.path.join(self.DATA_PATH, f'vid{self.VIDEO_NAME}.p')
        self.SAVE_DETECTIONS = os.path.join(self.DATA_PATH, f'detect{self.VIDEO_NAME}.p')
        self.SAVE_LANES = f'./data/lanes_detections{self.VIDEO_NAME}.csv' ###### Saves info about lanes
        
        self.process_every = process_every

        isContinue = self._check_existing()
        if isContinue:
            self._make_directories()
            print(self.CAM_VIEW)
            extractFrame(self.VIDEO_FILE, frameno = 0, dest_file = self.CAM_VIEW)
            self.cam_points, self.bird_points = self._get_reference_points()
            self._write_config()
        
    def _check_existing(self):
        """
        Returns:
            bool: if file exits and if user wants to continue
        """
        if Path(self.PARAM_FILE).exists():
            print("It looks like the directories, transformation matrix, and json file has already been set up for this video.")
            isContinue = input("Do you want to redo the setup? (y/n): ").lower().strip() == 'y'
            if not isContinue:
                print("Skipping setup")
                return False
        return True
        
    def _make_directories(self):
        Path(self.VIDEO_NAME).mkdir(parents=True, exist_ok=True)
        Path(self.DATA_PATH).mkdir(parents=True, exist_ok=True) 
        Path(self.IMAGE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.FRAME_FOLDER).mkdir(parents=True, exist_ok=True)
        print(f"Made directories for video {self.VIDEO_NAME}")
        
        return

    def _get_reference_points(self):
        """prompts user to select 4 corresponding points on 
        camera view and birds eye view

        Returns:
            np array, np array: user points selected
        """
        isHappy = False
        while not isHappy:
            plt.figure(0, figsize=(12, 10))
            plt.imshow(cv2.cvtColor(cv2.imread(self.CAM_VIEW), cv2.COLOR_BGR2RGB))
            plt.title("Camera View: \n \
                        Select 4 reference points (ie. Top left post, Bottom left sign, etc) \n\
                        Instructions: click once on the window to make it active. \n\
                        Left click to selct a point, right click to undo")
            cam_points = plt.ginput(4)
            time.sleep(1)
            plt.close()
            
            plt.figure(1, figsize=(12, 10))
            plt.imshow(cv2.cvtColor(cv2.imread(self.BIRD_VIEW), cv2.COLOR_BGR2RGB))
            plt.title("Bird View: \n \
                        Select 4 IDENTICAL reference points IN THE SAME ORDER \n \
                        (ie. Top Left, Top Right, Bottom Left, Bottom Right)")
            bird_points = plt.ginput(4)
            time.sleep(1)
            plt.close()
            
            plt.figure(0)
            plt.imshow(cv2.cvtColor(cv2.imread(self.CAM_VIEW), cv2.COLOR_BGR2RGB))
            plt.scatter([x[0] for x in cam_points], [y[1] for y in cam_points], c='r')
            plt.title("When ready to say y/n in terminal, close both windows")
            plt.figure(1)
            plt.imshow(cv2.cvtColor(cv2.imread(self.BIRD_VIEW), cv2.COLOR_BGR2RGB))
            plt.scatter([x[0] for x in bird_points], [y[1] for y in bird_points], c='r')
            plt.title("Do the points match corresponding locations and selection order? \n\
                        (There is another window under this one)")
            plt.show()

            isHappy = input("Are you happy with the points selected (y/n): ").lower().strip() == 'y'

        return np.float32(cam_points), np.float32(bird_points)

    def _make_map(self):
        # Creates a (maxtrix) mapping from camera view to bird's eye view
        M = cv2.getPerspectiveTransform(self.cam_points[:4], self.bird_points[:4])
        Minv = cv2.getPerspectiveTransform(self.bird_points[:4], self.cam_points[:4])
        return 
    
    # TODO: whats going on with the mask stuff
    # way to auto generate? or how to you make it
    def _process_mask(self):
        return
    
    # TODO: where do birdEye and latLonCoordinates come from? 
    # wouldnt they be the same?
    def _write_config(self):
        print(f"Writing jsonfile at {self.PARAM_FILE}")
        cam_view = cv2.imread(self.CAM_VIEW)
        bird_view = cv2.imread(self.BIRD_VIEW)
        jsonfile = '''{{
    "videoShape" : [{}, {}],
    "birdEyeViewShape" : [{}, {}],
    
    "cameraPoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]],
    "birdEyePoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]],
    
    "birdEyeCoordinates" : [[96, 30], [635, 1151]],
    "latLonCoordinates" : [[-122.660828, 45.743893], [-122.659864, 45.742497]]
}}'''.format(
            cam_view.shape[1], cam_view.shape[0], # videoShape
            bird_view.shape[1], bird_view.shape[0], # birdEyeViewShape
            
            int(self.cam_points[0][0]), int(self.cam_points[0][1]), # cameraPoints
            int(self.cam_points[1][0]), int(self.cam_points[1][1]),
            int(self.cam_points[2][0]), int(self.cam_points[2][1]),
            int(self.cam_points[3][0]), int(self.cam_points[3][1]), # cameraPointsEnd 
            
            int(self.bird_points[0][0]), int(self.bird_points[0][1]), # birdEyePoints
            int(self.bird_points[1][0]), int(self.bird_points[1][1]),
            int(self.bird_points[2][0]), int(self.bird_points[2][1]),
            int(self.bird_points[3][0]), int(self.bird_points[3][1]) # birdEyePointsEnd
        )

        with open(self.PARAM_FILE, 'w') as f:
            for line in jsonfile.split('\n'):
                f.write(line + '\n')
        
        return