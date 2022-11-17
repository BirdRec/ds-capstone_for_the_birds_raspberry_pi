######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Adapted: Friederike Thies, Egle Wahl, Raywant Kaur, Philipp Werden
# Date: 10/27/19 -> 11/2022
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# 
# The adaptation will add a save to dropbox option and a bird species classification.

# new packages
from __future__ import print_function
from classes.videostream_new import PiVideoStream
# Import packages
from classes.tempimage import TempImage
from classes.videostream import VideoStream
from imutils.video import FPS
import os
import argparse
import cv2
import numpy as np
import sys
import time
import datetime
import dropbox
import requests
import importlib.util
# new packages for picamera
import imutils
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera

# setup switch for sample model to custom model
use_custom_model = False

# Setup Dropbox 
# If you want to use dropbox, set this item to True, otherwise False
use_dropbox = True
# Adding the name of folder will directly add this to your main folder
# Recommendation: use your zip code so that geo information is available
your_base_path = "22589"
# Add Dropbox token
# In the 2_Dropbox_Guide.md it is drescribed how to obtain the tokens, 
# Follow the steps and use the creds here AFTER pulling this to your raspberrypi/webcam
your_app_key = "APP_KEY"
your_app_secret = "APP_SECRET"
your_oauth2_refresh_token = "REFRESH_TOKEN"
# Establish a connection:
# Check if dropbox us is enabled
if use_dropbox == True:
    # connect dropbox client
    client = dropbox.Dropbox(app_key = your_app_key,
            app_secret = your_app_secret,
            oauth2_refresh_token = your_oauth2_refresh_token)
    print("[SUCCESS] Dropbox accounted linked")

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default="Sample_TFLite_model")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
args = parser.parse_args()

# Initialize parameters
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
c = 0
lastUploaded = datetime.datetime.now()
min_upload_seconds = 3

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
print("[INFO] loading model...")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Camera stuff
# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] starting video stream from picamera...")
videostream = PiVideoStream(resolution=(imW,imH),framerate=32).start()
time.sleep(1)
fps = FPS().start()

# old: for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    # Set the current timestamp
    timestamp = datetime.datetime.now()

    # Grab frame from video stream
    frame1 = videostream.read()
    if frame1 is None:
        break
    frame1 = imutils.resize(frame1, width=1280)
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    # Coco model is the sample model these allocations work for it
    if use_custom_model == False:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    # Custom model: use these allocations to get all output details - change from tf1 to tf2
    else: 
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
        classes = interpreter.get_tensor(output_details[3]['index'])[0]     
        scores = interpreter.get_tensor(output_details[0]['index'])[0] 

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
            # Set timestamtp format for saving
            ts = timestamp.strftime("%Y-%m-%d_%X")

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            # See if enough time between uploads has passed
            if (timestamp - lastUploaded).seconds >= min_upload_seconds:
                # If label is "person" then upload
                if object_name in("bird","cat","Bird", "Cat", "Squirrel", "Raccoon"):
                    # Create temporary snapshot
                    t = TempImage()
                    cv2.imwrite(t.path, frame)
                    
                    # Upload temporary file to dropbox and cleanup temporary file
                    dropbox_path = "/{base_path}/{timestamp}_{object_name}.jpg".format(
                        base_path=your_base_path, timestamp=ts,object_name=object_name)
                    client.files_upload(open(t.path,"rb").read(),dropbox_path)
                    print("[UPLOADING to dropbox...] {}".format(ts))
                    t.cleanup()
                    
                    # Trigger IFTTT notification
                    #requests.post('https://maker.ifttt.com/trigger/bird_surveillance/with/key/c2nrke4lrR-FSkYsBS2EWX')
            
                    # Set last Upload to current time
                    lastUploaded=timestamp
            else:
                None
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    frame_display = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Object detector', frame_display)
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Update the FPS counter
    fps.update()

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Stop FPS thread
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up
cv2.destroyAllWindows()
videostream.stop()
