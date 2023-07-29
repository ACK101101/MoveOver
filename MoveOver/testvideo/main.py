import pandas as pd

from initialize import MakeParams
# from make_detections import MakeDetections

# setup paths and directories
# pathlib.Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True)
# pathlib.Path('./images/').mkdir(parents=True, exist_ok=True) #! Might not need 

### TODO: Checkpoint / customize what parts get run
### TODO: email psekula; where do the mask and coordinates come from?
### TODO: message Haley; possible to get coordinates and masks?
### ISSUE: LOOK OUT FOR WHERE params.json IS USED!

def main():
    """for each video in clips directory:
        create video specific folder that contains:
            config file
            camera, bird view, and mask**
            pickled video and detections
    """
    experiment_df = pd.read_csv("experimentInfo.csv")
    for _, video_info in experiment_df.iterrows():
        print(f'##### Video {video_info["VideoName"]} #####')
        MakeParams(video_info)
        # MakeDetections(VIDEO_NAME, keep_awake=False, process_every=2)
        
    return

main()
