import cv2
import numpy as np
import glob
import os
 
# Reference: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/

img_array = []
image_filenames = os.listdir('dot_track/Track_output/')

# Sort the filenames alphabetically
image_filenames.sort()

for filename in glob.glob('dot_track/Track_output/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('track_output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()