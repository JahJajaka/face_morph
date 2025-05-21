# Face Morphing Multiple Images Syncronized With Audio

All images in the folder morphed from one to another until the last one. The duration of morphing is syncronized with extracted beat structure.

# Execution

1. Place .mp3 file in the code folder. 
2. To extract beats run:

```python code/beat_extraction.py --input sigma_boy.mp3 --output beats.csv```

N.B: adjust --threshold parameter to get number of beats roughly equal number of transformations

3. Launch the command to create the video : 

```python code/__init__.py --folder aligned_images/ --beats code/beats.csv --output morphed_sequence.mp4```

Note that this will create temporary videos (```--tmpfolder```) and then combine them into one video (```--output```).

## How it works on multiple videos?
1. The program goes through all the images in ```--folder``` with a for loop
2. For each 2 images we apply what was done by Azmarie's original repo and this outputs a video. We store them in ```--tmpfolder```
3. We append to a text file named ```imageslist.txt``` the names of the videos like so : ```file '<filename_of_the_video>'```
4. After dealing with all images we can encode one big video using ```imageslist.txt``` and the right ffmpeg command:

```ffmpeg -f concat -safe 0 -i imageslist.txt -c copy output.mp4```
5. If process was interupted for some reason it will be resumed from the latest found segment of video

# Details to call code/__init__.py

- ```--folder``` : The folder with all images to morph
- ```--duration``` : The duration of morphing from one image to the other.
- ```--frame``` : The frame rate of the encoding.
- ```--output``` : Final video path.
- ```--tmpfolder``` : Folder to store intermediate videos.
- ```--beats``` : Path to csv file with extracted beats
- ```--beat_interval``` : Use only every Nth beat instead of each one
