import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'..')

import config
import pathlib

from make_detections import MakeDetections
from moveoverlib.helper import ImageEncoder, create_box_encoder

# setup paths and directories
pathlib.Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/').mkdir(parents=True, exist_ok=True) #! Might not need 

### TODO: Checkpoint / customize what parts get run
### TODO: email psekula; where do the mask and coordinates come from?
### TODO: message Haley; possible to get coordinates and masks?
### ISSUE: LOOK OUT FOR WHERE params.json IS USED!

# set videofiles
VIDEO_NAME = '4'
MakeDetections(VIDEO_NAME, keep_awake=False, process_every=2)

# TOPOINT = 'BL10' ## Bottom Left
# image_encoder = ImageEncoder(config.ENCODER_PATH, config.ENCODER_INPUT_NAME, config.ENCODER_OUTPUT_NAME)
# encoder = create_box_encoder(config.ENCODER_PATH, batch_size=32)

