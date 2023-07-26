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
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

# TODO: shorten imports
# TODO: why use an encoder if the detections are already done
# TODO: what is TOPOINT
# TODO: resize isnt used?
# TODO: why use socket in main loop
# TODO: dont understand tracker predict and update

# TODO: make skip_every consistent with make_detections.py

class ReadDetections():
    def __init__(self, vidname, save_log=None, max_cos=0.2, nn_budget=100, skip_first=0):
        # TODO: setup paths to video, params, aerial, mask
        self.VIDEO_NAME = vidname
        self.detections = pickle.load(
            open(f'{config.DATA_PATH}/detect{self.VIDEO_NAME}.p',
                 'rb')
            )
        self.FRAME_FOLDER = os.path.join(config.DATA_PATH, 
                                         f'frames_raw{self.VIDEO_NAME}/')  
        Path(self.FRAME_FOLDER).mkdir(parents=True, exist_ok=True)
        self.VIDEO_FILE = pickle.load(open(f'{config.DATA_PATH}/vid{self.VIDEO_NAME}.p', 'rb'))
        self.SAVE_LANES = f'./data/lanes_detections{self.VIDEO_NAME}.csv' ###### Saves info about lanes
        self.SAVE_LOG = save_log #### Saves logs with all detected objects (path to file or none)
        
        # init image encoder
        self.ENCODER_PATH = config.ENCODER_PATH
        self.ENCODER_INPUT_NAME = config.ENCODER_INPUT_NAME
        self.ENCODER_OUTPUT_NAME = config.ENCODER_OUTPUT_NAME
        self.BATCH_SIZE = config.ENCODER_BATCH_SIZE
        self.TOPOINT = 'BL10' # Bottom Left
        self.encoder = create_box_encoder(self.ENCODER_PATH, batch_size=self.BATCH_SIZE)
        
        # init tracker detection trajectories
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cos, nn_budget)
        self.tracker = Tracker(self.metric)
        
        # init map between camera and aerial view
        # TODO: make paths specific to video name
        self.params = Parameters()
        self.params.generateParameters('./params.json')
        self.mymap = Map('./images/SkyView.jpg', './icons_simple.json', self.params)
        
        # init lane detection with mask
        self.lanes_controller = Lanes('./images/mask.png', params=self.params)
        
        # How many seconds in the beginning should be skipped
        self.SKIP_FIRST = skip_first
        
        self._process_detections()
    
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

                #! what is the trackedObject - bb?
                obj = track.trackedObject

                if obj is not None:
                    if obj.color is None:
                        obj.color = (randint(0, 255), randint(0, 255), randint(0, 255))                        
                    plotbox = obj.bboxes[-1]
                    plotbox.trackId = track.track_id
                    plotboxes.append(plotbox)
                    plotcolors.append(obj.color)
                    objects.append(obj)
                
                    #! need to look further into below class
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
        if len(objects) > 0: 
            logfile = open('./logs/{}'.format(self.SAVE_LOG, 'w'))
            for obj in objects:
                #! error with True  
                line = '{},{},{}'.format(frame_num-1,time(),obj.getParams(asCsv=True))                     
                print(line,file=logfile)   
        
        return
    
    def _process_detections(self):
        # opens video file, gets metadata about video
        cap = cv2.VideoCapture(self.VIDEO_FILE) 
        FRAMES_SEC = cap.get(cv2.CAP_PROP_FPS)
        # TODO: change to list? arg for crop
        CROP_VID = False
        VID_LEFT = 0
        VID_RIGHT = 1920
        VID_UP = 0

        objects = []
        frame_num = 0
        start_time = time()
        
        # skips first k seconds
        while frame_num < self.SKIP_FIRST * FRAMES_SEC:
            _, image = cap.read()
            frame_num += 1

        #! why use a socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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
                if frame_num not in self.detections:
                    continue
                
                # read detections
                boxes, scores, classes = self.detections[frame_num] 
                
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
