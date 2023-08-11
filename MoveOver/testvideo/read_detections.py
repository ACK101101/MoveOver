import cv2
import os
import sys
import pickle
import socket
import config
import numpy as np
from random import randint 
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
from cav.lanes import Lanes
from moveoverlib.helper import ImageEncoder, create_box_encoder
from cav.parameters import Parameters
from cav.visualization import Map, plotBoxes
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

# TODO: when generating params, where do "birdEyeCoordinates" and "latLonCoordinates" come from
# TODO: why use an encoder if the detections are already done
# TODO: what is TOPOINT
# TODO: resize isnt used?
# TODO: why use socket in main loop
# TODO: dont understand tracker predict and update

# TODO: make skip_every consistent with make_detections.py

class ReadDetections():
    def __init__(self, Paths, save_log=None, max_cos=0.2, nn_budget=100, skip_first=0):
        # TODO: setup paths to video, params, aerial, mask 
        self.SAVE_LOG = save_log #### Saves logs with all detected objects (path to file or none)
        self.TOPOINT = 'BL10' # Bottom Left
        
        # init image encoder
        self.ENCODER_PATH = config.ENCODER_PATH
        self.ENCODER_INPUT_NAME = config.ENCODER_INPUT_NAME
        self.ENCODER_OUTPUT_NAME = config.ENCODER_OUTPUT_NAME
        self.BATCH_SIZE = config.ENCODER_BATCH_SIZE
        
        self.encoder = create_box_encoder(self.ENCODER_PATH, batch_size=self.BATCH_SIZE)
        
        # init tracker detection trajectories
        self.metric = NearestNeighborDistanceMetric("cosine", max_cos, nn_budget)
        self.tracker = Tracker(self.metric)
        
        # init map between camera and aerial view
        # TODO: make paths specific to video name
        self.params = Parameters()
        self.params.generateParameters(Paths.PARAM_FILE)
        # TODO: include icon path in init script?
        self.mymap = Map(Paths.BIRD_VIEW, './icons_simple.json', self.params)
        
        # init lane detection with mask
        # TODO: make paths specific to video mask
        self.lanes_controller = Lanes('./images/mask.png', params=self.params)
        
        # How many seconds in the beginning should be skipped
        # TODO: make consistent with make_detections.py
        self.SKIP_FIRST = skip_first
        
        self._process_detections(Paths)
    
    def _track_detections(self, bgr_image, boxes, scores, classes):
        for box in boxes:
            if self.TOPOINT != 'BC': box.setToPoint(self.TOPOINT)
            #! TODO: don't understand
            box.updateParams(self.params)
        
        boxes_array = np.array([[box.xLeft, box.yTop, 
                                box.xRight - box.xLeft, 
                                box.yBottom - box.yTop] for box in boxes])
        
        #! need to understand encoder
        features = self.encoder(bgr_image, boxes_array)
        detections = []
        
        #! reading in from detections file, makes Detection object
        for box, score, objClass, f_vector in zip(boxes, scores, classes, features):
            detection = Detection([box.xLeft, box.yTop, 
                                    box.xRight - box.xLeft, 
                                    box.yBottom - box.yTop], #BBox
                                    score, f_vector,objClass
                                )
            detection.bbox = box
            detections.append(detection)

        #! updates detections list directly?
        self.tracker.predict()
        self.tracker.update(detections)    
        
        return
    
    def _save_detections(self, image, frame_num, logfile_lanes):
        plotboxes = []
        plotcolors = []
        objects = []

        if len(self.tracker.tracks) >= 1:
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update >= 1:
                    continue

                obj = track.trackedObject

                if obj is not None:
                    if obj.color is None:
                        obj.color = (randint(0, 255), randint(0, 255), randint(0, 255))                        
                    plotbox = obj.bboxes[-1]
                    plotbox.trackId = track.track_id
                    plotboxes.append(plotbox)
                    plotcolors.append(obj.color)
                    objects.append(obj)
                
                    lane = self.lanes_controller.addObject(obj)
                    log_line = '{},{},{}'.format(frame_num-1, lane, 
                                                    obj.getParams(asCsv=True, speedLookback = 10)
                                                )
                    print(log_line, file=logfile_lanes)                              

            #! plots bb and saves                                             
            if len(plotboxes) >= 1:
                vid = plotBoxes(image, plotboxes, colors=plotcolors)
            else:
                vid = image.copy()
            cv2.imwrite(os.path.join(self.FRAME_FOLDER, 'im_{}.jpg'.format(str(frame_num-1).zfill(6))), vid)
        
        #! can clean up structure
        #! save objects
        if self.SAVE_LOG and len(objects) > 0: 
            logfile = open('./logs/{}'.format(self.SAVE_LOG, 'w'))
            for obj in objects:
                line = '{},{},{}'.format(frame_num-1,time(),obj.getParams(asCsv=True))                     
                print(line,file=logfile)   
        
        return
    
    def _process_detections(self, Paths):
        # opens video file, gets metadata about video
        cap = cv2.VideoCapture(Paths.VIDEO_BINARY) 
        FRAMES_SEC = cap.get(cv2.CAP_PROP_FPS)
        
        # TODO: change to list? arg for crop
        CROP_VID = False
        VID_LEFT = 0
        VID_RIGHT = 1920
        VID_UP = 0

        objects = []            # TODO: not used
        frame_num = 0
        start_time = time()
        
        # skips first k seconds
        while frame_num < self.SKIP_FIRST * FRAMES_SEC:
            _, image = cap.read()
            frame_num += 1
        
        # open lane log file to write to
        logfile_lanes = open('{}'.format(self.SAVE_LANES), 'w')

        while cap.isOpened():
            # time since starting and stats
            curr_time = time() - start_time
            sys.stdout.write('{} frames done in {:.1f} seconds ({:.2f} frames/sec)    \r'.format(
                frame_num, curr_time, frame_num/curr_time))         
            
            _, image = cap.read()
            frame_num += 1
                    
            # crops the video
            if CROP_VID: image = image[VID_UP:, VID_LEFT:VID_RIGHT, :]
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # checks if frame in detections; if not, skip iter
            if frame_num not in Paths.SAVE_DETECTIONS:
                continue
            
            # read detections
            boxes, scores, classes = Paths.SAVE_DETECTIONS[frame_num] 
            
            if len(boxes) >= 1:
                self._track_detections(bgr_image, boxes, scores, classes)           
            else:
                self.tracker.predict()
                
            self._save_detections(image, frame_num, logfile_lanes)                   
            
        #! end stats video                  
        start_time = time() - start_time                             
        print('\n\n{} frames done in {:.1f} seconds ({:.2f} frames/sec)'.format(
            frame_num, start_time, frame_num/start_time))                             
        cap.release()
        
        return
