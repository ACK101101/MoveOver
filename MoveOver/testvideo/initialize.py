### Importing packages
import sys
# Adds parent directory, MoveOver, to the path 
sys.path.insert(0,'..')

import cv2
import pathlib 
import time
import pickle
import config
import numpy as np
from moveoverlib.functions import *
import matplotlib.pyplot as plt

# TODO: write useful variables to config file
#       OR
#       Make experiment class for each varible

# TODO: add check for existing config/param file, skip if so

class MakeParams():
    """ generate config file
        setup file and directory names
        prompt user to select reference points to generate map
    """
    def __init__(self, video_info):
        # Provided example video
        self.VID_NAME = video_info["VideoName"]
        self.VIDEO_FILE = f'./clips/{self.VID_NAME}.mp4'
        self.CAM_VIEW = f'./images/cam_view{self.VID_NAME}.jpg'         # setup file name
        self.BIRD_VIEW = f'./images/{video_info["BirdView"]}'           # already exists     
        
        self._make_directories()
        
        extractFrame(self.VIDEO_FILE, frameno = 0, dest_file = self.CAM_VIEW)
        
        # TODO: not used; save or delete?
        self.cam_points, self.bird_points = self._get_reference_points()
        
        self._write_config()
        
    def _make_directories(self):
        pathlib.Path(self.VID_NAME).mkdir(parents=True, exist_ok=True)
        pathlib.Path('./data/').mkdir(parents=True, exist_ok=True) 
        pathlib.Path('./images/').mkdir(parents=True, exist_ok=True)
        print(f"Made directories for video {self.VID_NAME}")
        
        return

    def _get_reference_points(self):
        isHappy = False
        while not isHappy:
            plt.figure(0, figsize=(7, 5))
            plt.imshow(cv2.cvtColor(cv2.imread(self.CAM_VIEW), cv2.COLOR_BGR2RGB))
            plt.title("Camera View: \n Select 4 reference points (ie. Top Left, Top Right, Bottom Left, Bottom Right)")
            cam_points = plt.ginput(4)
            time.sleep(1)
            plt.close()
            
            plt.figure(1, figsize=(7, 5))
            plt.imshow(cv2.cvtColor(cv2.imread(self.BIRD_VIEW), cv2.COLOR_BGR2RGB))
            plt.title("Bird View: \n Select 4 IDENTICAL reference points IN THE SAME ORDER \n (ie. Top Left, Top Right, Bottom Left, Bottom Right)")
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
            plt.title("Do the points match corresponding locations and selection order?")
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
    
    # TODO: what params are actually used later?
    # I don't think the coordinates are used. maybe good for storage? idk
    # could save as not a json file
    def _write_config(self):
        cam_view = cv2.imread(self.CAM_VIEW)
        bird_view = cv2.imread(self.BIRD_VIEW)
        jsonfile = '''{{
            "videoShape" : [{}, {}],
            "birdEyeViewShape" : [{}, {}],
            
            "cameraPoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]],
            "birdEyePoints" : [[{}, {}], [{}, {}], [{}, {}], [{}, {}]]
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

        with open(f'./{self.VID_NAME}/params.json', 'w') as f:
            for line in jsonfile.split('\n'):
                f.write(line + '\n')
        
        return