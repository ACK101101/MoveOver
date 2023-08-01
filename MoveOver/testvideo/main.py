import pandas as pd

from initialize import Experiment
from make_detections import MakeDetections
from read_detections import ReadDetections

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
        Paths = Experiment(video_info)
        MakeDetections(Paths, keep_awake=False, process_every=2)
        
    return

main()
