import pandas as pd

from initialize import Experiment
from make_detections import MakeDetections
from read_detections import ReadDetections
from analyze_detections import AnalyzeDetections

def getNumeric(prompt):
    while True:
        try:
            res = int(input(prompt))
            break
        except ValueError:
            print("Numbers only please!")
    return res

def main():
    """for each video in clips directory:
        create video specific folder that contains:
            config file
            camera, bird view, and mask**
            pickled video and detections
    """
    experiment_df = pd.read_csv("experimentInfo.csv")
    for _, video_info in experiment_df.iterrows():
        if video_info["VideoName"] == "testvid":
            print(f'##### Video {video_info["VideoName"]} #####')
            process_every = getNumeric("Process every kth frame? Input an integer > 0. \
                                    1 means processing every frame, 2 every other frame, etc. :")
            Paths = Experiment(video_info, process_every)
            MakeDetections(Paths, keep_awake=False)
            ReadDetections(Paths)
            
        
    return

main()
