## Face recognition
### Assigment of Key Topics of Artificial Intelligence OU, Netherlands 2024

Ralph Depondt
Evert van Kammen
Raymond

Maximum Python version to use is 3.11, mediapipe won't install with version 3.12.

    pip install mediapipe
    pip install opencv-python
    pip install face-recognition
    pip install torch torchvision torchaudio 
    pip install pandas
    pip install numpy
    pip install matplotlib

# Face Recognition Project README

## Introduction
This project consists of several Python scripts for encoding images, recognizing faces in videos, and conducting experiments. 
Additionally, there is a shared module `shared.py` that contains utility functions, needed for the entire project.

## Usage

### Shared Module (`shared.py`)
The shared.py module contains various utility functions that can be used across different parts of the project.

### Encode Images
We designed two ways to encode images and categorize the actors from the scenes:
- Taking pictures from the internet and organising them per actor.
- Taking screenshots from the video clip and tell to which actor the found face belongs

#### Internet pictures
To encode the images from the internet, pictures were downloaded and saved in a directory. 
The subdirectories contain the names of the actor a picture belongs to.

To encode the pictures, use encoder_main.py. (There is also a method made for this in shared.py)
There are for variables you should think about:
- image_path: The directory the images are located
- encode_model: The way to encode the images ('hog' or 'cnn'), where cnn is more accurate but requires more performance
- number_of_images: max number of images to encode from a subdirectory (None, will encode all images in the subdirectory)
- pickle_output_path: output path of the encoding

After running encoder_main.py, a pickle file is created, which can be used to run the face recognition.

#### Mediapipe [TODO- EVERT]

### Experiments
For each experiment, create a directory with the name of the experiment that you want to conduct.
Also for each experiment, an experiment.py is created which creates graphs which could be used in reports.

#### Run face recognition
After running the encoding, it is possible to load a video clip and recognize actors in the clip.

To run the face-recognition use: main.py (there is also a feature made in shared.py)
These are the variables you could use and are relevant for this project:
- video_file: location of the video clip
- encodings_file: location of the encodings file
- process_every_nth_frame: Check to make it possible to skips frames in processing (1, if you want to process every frame)
- experiment_results: location where you want to store the results of the experiment
- show_display: If you want to show the clip during processing
- desired_tolerance: the tolerance of how sure the program is that an actor is recognized
- desired_width: Width of the frame to process, make this smaller for faster performance, but less accurate results
- desired_model: Model to do the calculations: ('hog' or 'cnn'), where cnn is more accurate
- upsample_times: Times to upsample the frame (0, for no upsampling)
- sample_probability: The chance you want to process a frame
- save_images: True if you want to save the image of a frame that was processed.

After running main.py, a .txt file is created, with can be used for the experiments.


Practical links on OpenCV and face recognition: 

https://youtu.be/5yPeKQzCPdI?si=eZse0pvb3CvGrhmF

https://youtube.com/playlist?list=PLzMcBGfZo4-lUA8uGjeXhBUUzPYc6vZRn&si=yMqLKPz-F78l662r

