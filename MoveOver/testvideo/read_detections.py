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

class ReadDetections():
    def __init__(self):
        self.ENCODER_PATH = config.ENCODER_PATH
        self.ENCODER_INPUT_NAME = config.ENCODER_INPUT_NAME
        self.ENCODER_OUTPUT_NAME = config.ENCODER_OUTPUT_NAME
        self.BATCH_SIZE = config.ENCODER_BATCH_SIZE
        pass
    
    def _process_detections(self):
        TOPOINT = 'BL10' # Bottom Left
        image_encoder = ImageEncoder(self.ENCODER_PATH, 
                                     self.ENCODER_INPUT_NAME, 
                                     self.ENCODER_OUTPUT_NAME)
        encoder = create_box_encoder(self.ENCODER_PATH, batch_size=self.BATCH_SIZE)
        
        #! max distance from kth centroid?
        max_cosine_distance = 0.2
        #! only check first k distances per centroid?
        nn_budget = 100
        #! seems to init some sort of nn object
        #! need to move down
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        
        params = Parameters()
        params.generateParameters('./params.json')
        mymap = Map('./images/SkyView.jpg', './icons_simple.json', params)
        
        #! unsure, need to run to see how it works as lanes.py is poorly documented
        lanes_controller = Lanes('./images/mask.png', params=params)
        
        #! setting up paths for main loop
        #! can clean up ordering to make more readable
        #! don't understand save_detections, creates empty file then reads from it 
        SAVE_DETECTIONS = f'{config.DATA_PATH}/detections.p'
        FRAME_FOLDER = os.path.join(config.DATA_PATH, 'frames_raw/')
        VIDEO_FILE = pickle.load(open(f'{config.DATA_PATH}/videopath.p', 'rb'))
        print ('Video path:', VIDEO_FILE)

        Path(FRAME_FOLDER).mkdir(parents=True, exist_ok=True)

        save_detections = pickle.load(open(SAVE_DETECTIONS,'rb'))
        
        SAVE_LOG = None #### Saves logs with all detected objects (path to file or none)

        #! can try both; what info?
        SAVE_LANES = None ###### Saves info about lanes
        SAVE_LANES = './data/lanes_detections.csv' ###### Saves info about lanes

        SKIP_FIRST = 0 # How many seconds in the beginning should be skipped
        
        #! opens video file, gets metadata about video
        cap = cv2.VideoCapture(VIDEO_FILE) 
        FRAMES_SEC = cap.get(cv2.CAP_PROP_FPS)
        VIDEO_X = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        VIDEO_Y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

        #! hyperparams for detection
        MAX_BOXES_TO_DRAW = 100       #! not 100% sure, max bb per frame?
        MIN_SCORE_THRESH = 0.5        #! confidence thresh for classification?
        IOU_COMMON_THRESHOLD = 0.50   #! same as above?
        NOT_DETECTED_TRHESHOLD = 1    #! not sure

        MAPSHAPE = mymap.getMap().shape
        print ('Y dimension of map is {:.3f} larger than Y dimension of the video'
            .format(MAPSHAPE[0] / VIDEO_Y))

        #! downsampling?
        MAP_RESIZE = 3

        print ('Y dimension of map is {:.3f} larger than Y dimension of the video. Size of the map is reduced {} times.'
            .format(MAPSHAPE[0] / VIDEO_Y, MAP_RESIZE))


        FINAL_X = VIDEO_X + int(MAPSHAPE[1] / MAP_RESIZE)
        FINAL_Y = max(VIDEO_Y, int(MAPSHAPE[0] / MAP_RESIZE))

        print ('Video size: [{}, {}], Final size: [{}, {}]'
            .format(VIDEO_X, VIDEO_Y, FINAL_X, FINAL_Y))

        RESIZE = False                #! not used

        #! unclear, but pixel coords for video?
        CROP_VID = False
        VID_LEFT = 0
        VID_RIGHT = 1920
        VID_UP = 0
        
        #! why reopen? seems like didn't actually modify the video above
        cap = cv2.VideoCapture(VIDEO_FILE) 

        #! specifies video codec but not sure what XVID is
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = None              #! not used

        objects = []

        results = []            #! not used
        colors = {}             #! not used

        #! related to NN and BB but not sure how
        tracker = Tracker(metric)

        nr_skipped = 0          #! not used
        i = 0
        t = time()

        #! why use a socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # open lane log file to write to
            if SAVE_LANES is not None:
                logfile_lanes = open('{}'.format(SAVE_LANES), 'w')

            while cap.isOpened():
                # time since starting and stats
                t2 = time() - t
                sys.stdout.write('{} frames done in {:.1f} seconds ({:.2f} frames/sec)    \r'.format(
                    i, t2, i/t2))                   
                
                frame_timeStamp = i/FRAMES_SEC      #! not used
                
                #! gets next frame; not sure what ret is maybe metadata
                ret, image = cap.read()

                # skips first k frames
                if i < SKIP_FIRST * FRAMES_SEC:
                    i += 1
                    continue
                        
                # crops the video
                if CROP_VID:
                    image = image[VID_UP:, VID_LEFT:VID_RIGHT, :]
                
                #! is save_detections a list of idxs := frames
                if i+1 not in save_detections:
                    break
                
                #! even more confused as to what save_detections is
                #! reading in predone detections per frame?
                ###!!! what object is being saved in the save_detections file
                boxes, scores, classes = save_detections[i+1] 
                
                if len(boxes) >= 1:
                    for box in boxes:
                        if TOPOINT != 'BC':
                            box.setToPoint(TOPOINT)         #! find in box class?
                            
                        box.updateParams(params)
                    
                    #! what are these bb? for
                    boxes_array = [[box.xLeft, box.yTop, box.xRight - box.xLeft, box.yBottom - box.yTop] for box in boxes]
                    boxes_array = np.array(boxes_array)
                    
                    #! need to understand encoder
                    bgr_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    features = encoder(bgr_image, boxes_array)
                    detections = []

                    #! reading in from save_detections file, makes Detection object
                    for box, score, objClass, f_vector in zip(boxes, scores, classes, features):
                        detection = Detection(
                            [box.xLeft, box.yTop, box.xRight - box.xLeft, box.yBottom - box.yTop], #BBox
                            score, f_vector,
                            objClass
                        )
                        detection.bbox = box
                        detections.append(detection)

                    #! updates detections list directly?
                    tracker.predict()
                    tracker.update(detections)                
                
                #! if no boxes, what is being used to predict??
                else:
                    tracker.predict()
                    
                plotboxes = []
                plotcolors = []
                objects = []

                if len(tracker.tracks) >= 1:
                    for track in tracker.tracks:
                        #! huh? why not use break structure
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
                            
                            if SAVE_LANES is not None:
                                #! need to look further into below class
                                lane = lanes_controller.addObject(obj)
                                if SAVE_LANES is not None:
                                    log_line = '{},{},{}'.format(i, lane, obj.getParams(asCsv=True, speedLookback = 10))
                                    print(log_line,file=logfile_lanes)                              

                    #! plots bb and saves                                             
                    if len(plotboxes) >= 1:
                        vid = plotBoxes(image, plotboxes, colors=plotcolors)
                    else:
                        vid = image.copy()
                    cv2.imwrite(os.path.join(FRAME_FOLDER, 'im_{}.jpg'.format(str(i).zfill(6))), vid)

                #! can clean up structure
                #! save objects
                if len(objects) > 0:        
                    if SAVE_LOG is not None:
                        logfile = open('./logs/{}'.format(SAVE_LOG, 'w'))
                        for obj in objects:
                            #! error with True  
                            line = '{},{},{}'.format(i,time(),obj.getParams(asCsv=true))                     
                            print(line,file=logfile)                    
                                        
                i = i+1
                        
        #! end stats video                  
        t = time() - t                             
        print('\n\n{} frames done in {:.1f} seconds ({:.2f} frames/sec)'.format(
            i, t, i/t))                             
        cap.release()
