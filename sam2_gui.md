# SAM 2 GUI 

## Idea

The current implemenetation of SAM 2 for vocal tract segmentation as seen in the SAM2_childrensdata... notebook is roughly as follows:

A single video file is loaded, and an initial frame is selected for prompting the model. 

The user goes through multiple iterations of setting points in the image manually using coordinates with positive and negative prompt points used. This guides the model to segment certain areas of the image. 

Once the user goes through these iterations and settles on a final set of segmented regions, these get propagated through the entire video. 

These sets of regions can be re-used across multiple videos for a single speaker, though the script does not have a for loop for doing this automatically. 

I want to create a GUI that allows one to choose a speaker from a /data directory, select any file for that speaker, choose an intial frame within a video using a scroll/play bar and view live the model iterations of segmentation as one clicks to create points for positive/negative feedback. The final regions can be saved and then also used to automatically save the masks for all videos for a given speaker. 

## Basic approach

1. The GUI can be roughly based on that in /Users/seanfoley/Desktop/prompt/src/roi_tool/roi_editor.py

2. The user runs the GUI; selects a speaker from a drop-down menu. The speaker can also select video files for that speaker using a drop-down menu. 

3. Use a config file to specify data_dir and other potential parameters. 

4. First, an initial frame is needed for segmentation. When viewing a video, the user can scroll through the video and when they land on a good frame, they can click "Set as initial frame". This freezes the view on that frame. 

5. Given this initial frame, the user can select to click a positive or negative star icon for prompting the model segmentation. The model will update with each new added point. The user should be able to delete or clear all points for a region too. 

6. The user can add new regions for segmentation, each region has their own set of points and name. 

7. After the regions are set, they can be exported and saved, with the option to be loaded into the GUI later for editing. 

8. A new script is needed to then load in the regions and propagate through all videos in the spk_dir, saving the resulting masks timeseries and also saving overlay videos for diagnostics. This needs to be done in a memory efficient way, e.g. only propagate for so many video frames at a time and then concatenate, and allow for multi-GPU. 