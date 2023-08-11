# Move Over Modified
My main github: ACK101101

This was my summer effort to review the existing code ported over to the Move Over project, diagnosing quality of code and identifying errors/issues. The goal is to get this code ready for easy 3rd party use. It was not close to that stage and as of now still needs a lot of work. Below I will outline how the existing code works, the major issues with the existing code, the steps needed to fix them, as well as my modifications and additions to the code.

## Vocabulary
- satellite view, top down view, birds eye view := 
        
        Image taken from above, like from Google Earth 
        
        See example in testvideo/images/I-35_SB.PNG
- transformation matrix := See [cv2.getPersepctiveTransform()](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga8c1ae0e3589a9d77fffc962c49b22043)
        
        TLDR: have two images of the same subject from different angles. Select 4 points on the object in image 1. Select the same 4 points on the object in image 2. The function generates a mapping of pixel coordinates from image 1 to image 2
- mask :=
        
        Summary: This mask is used to determine if a detected vehicle Moved Over and/or slowed down. 

        There are 3 sections: before Move Over zone, Move Over zone, and after Move Over zone (this requires the user to determine where the Move Over zone is); this is used to determine slow down. 

        Each section split into two subsections: Move Over lane and other lanes; this is used to Move Over. The previous developer used the 3rd party software Gimp to draw the masks.

        See example in testvideo/images/mask.png
- reference points, landmark points := 

        Points on the subject of an image that can be easily found on another image of the same subject at a different angle. Used for transformation matrix.
- representative point :=

        A point in the bounding box of a detected vehicle is chosen tho represents the location of the vehicle. This makes it easier to transform with the matrix, as well as easier to determine what mask zone it is in.
        If the video is in the middle of a two-way highway facing straight forward, the a representative point the works well will be in the center of the bounding box. However, since most cameras are at an angle, the representative point needs to be skewed from the center of the bounding box so that it is still in the middle of the detected vehicle.

## Pipeline
This is referencing th existing Python notebooks under MoveOver/example
1. A json file is created with parameters that are used for later steps. It contains video metadata, reference points for a transformation matrix, coordinates, and a mask validity check.

2. The video is run through the detection model, producing bounding boxes around detected vehicles. Theres are saved in a pickle file.

3. This is where things get more confusing and obfuscated. The video is processed frame by frame. 
The bounding boxes associated with each frame are represented by a single point each; the representative points, frame, and mask are then projected using the transformation matrix to a birds eye view. Think of this as viewing everything happening on the road from above looking down - this makes it easier to determine where the detected vehicles are on the road when compared to a more street-level camera. 
A tracker is used to string together detections of the same vehicle across frames, so that it is organized under one trajectory. 
The trajectory is then analyzed with respect to the lane mask, recording the events from zone to zone in the mask. It records what lane they were in and if they Moved Over, saving the data in a csv.

4. This code is even more confusing and obfuscated. The lane detections from the previous part are used to determine if a detected vehicle was subject to the Move Over law, if it could Move Over, and if it did Move Over and/or slow down. This analysis is saved in a csv, and a video is generated with bounding boxes overlayed.

5. This code is for cleaning up and exporting the data for final use and analysis.

Important note: Most of my work involved investigating and modifying steps 1-3 

## Issues
### Meta Issues
1. The primary issue is the organization of the code. All the original code (found in MoveOver/example/) is written in Python notebooks:
        
        a. This is good for quick testing or presnting code that is already clean and neatly wrapped up, but the current state one of messy-middle-of-development. 

        b. There is little documentation and object oriented structuring, making reading, understanding, modifying, and contributing to the codebase extremely difficult.
2. The current implementation requires selecting corresponding landmark points in both the frame of the video and the satellite image. This limits the types of videos that can be easily compatible with the software, as it must have identifiable landmarks from both the camera view and satellite view.

### Technical Issues
The current implementation requires the user to have many manually performed prerequisites that are not specified:
        
        a. Have Latitude and Longitude geographic coordinates of the area that the video is capturing.

        b. Use the latLon coords to grab a satellite top-down image of the road; I used Google Earth.

        c. Pair 4 landmark points in both the frame of the video and the satellite image (making 8 total points, 4 pairs of 2). 

        d. Get the latLong coords of 2 landmark points from the satellite view. These 2 points must be sufficiently far apart because this is used to determine the distance and scale of the image. This is used later to derive speed.

        e. The landmark points selected in section c. are used to make a transformation matrix using cv2. This matrix maps points between the video POV and the satellite POV. To make the map, it requires reference points, which are the landmark points selected above.

        f. Make a mask of the road from the video POV. This mask is used to determine if a detected vehicle Moved Over and/or slowed down. 

        g. Skewing the representative point of the bounding boxes. Because of the different angles of video recording, the representative point might need to be manually adjusted to accurately represent the location of the detected vehicle

        h. The model is computationally expensive; with GPU it runs ~3/4 fps, without ~0.2/0.25 fps. As such, it is too slow 

## Modifications
These are the changes I made to the code:

1. I restructured and cleaned up notebooks 1, 2, and most of 3

2. I changed the pipeline and file structure so that multiple experiments can be run in succession and experiment output is saved in a separate directory, making post-experiment analysis a bit easier

3. I built a tool to help the user make the transformation matrix by prompting the user to select the landmark points using matplotlib


## Proposed Next Steps
1. Finish Object Oriented Programming restructuring of the notebooks

2. Automatic satellite image grabber using user provided geocoordinates

3. Add distance point selection to the transformation matrix tool created

4. Integrated masking tool

5. Automatic bounding box representative point choice based on angle

6. Use huggingface for ML usage

7. Make vehicle detection dataset used for finetuning the original model publicly available

## Misc
### Coordinates
This is the geographic coordinate information given by our joint effort with the KC Scout. Locations information of each video in testvideo/clips is in the upper left. For US-50 EB, I could not find the location on Google Maps due to a lack of landmarks.

• I-470 WB Past View High Dr (73): 38.937211, -94.457856

        Clip: 1, 4
        38°56'14.0"N 94°27'28.3"W - Google Maps

• I-435 SB At 23rd St (1030): 39.081339, -94.491155

        Clip: 3-01, 3-02, 
        39°04'52.8"N 94°29'28.2"W - Google Maps

• US-50 EB At 291 NB (1044): 38.902667, -94.362928
        
        Clip: 5, 6, 7

• I-35 SB At Shawnee Mission Pkwy (12): 39.0124, -94.6934

        Clip: 12, 13, 15
        39°00'44.6"N 94°41'36.2"W - Google Maps