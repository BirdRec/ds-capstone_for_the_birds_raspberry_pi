######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Adapted: Friederike Thies, Egle Wahl, Raywant Kaur, Philipp Werdenbach-J.
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

# Setup Dropbox 
# If you want to use dropbox, set this item to True, otherwise False
use_dropbox = True
# when saving items to your dropbox, we recommend to use your zip code so that geo information is available
your_base_path = "Apps/22589"
# Add Dropbox token
# In the 2_Dropbox_Guide.md it is drescribed how to obtain the tokens, 
# Follow the steps and use the creds here AFTER pulling this to your raspberrypi/webcam
your_app_key = "APP_KEY"
your_app_secret = "APP_SECRET"
your_oauth2_refresh_token = "REFRESH_TOKEN"

# Check if dropbox us is enabled
if use_dropbox == True:
    # connect dropbox client
    client = dropbox.Dropbox(app_key = your_app_key,
            app_secret = your_app_secret,
            oauth2_refresh_token = your_oauth2_refresh_token)
    print("[SUCCESS] Dropbox accounted linked")

# Define and parse input arguments for object detection model (od) and image classification model (cl)
parser = argparse.ArgumentParser()
# OD Model:
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default="/models/object_detection")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='coco_ssd_mobilenet_v1_1_0_quant_2018_06_29.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='/models/object_detection/labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
# CL Model:
parser.add_argument('--modeldir_cl', help='Folder the .tflite file is located in',
                    default="/models/image_classification")
parser.add_argument('--graph_cl', help='Name of the .tflite file, if different than detect.tflite',
                    default='MobileNetV2_Kaggle_p150_e10_dr-quant.tflite')
parser.add_argument('--labels_cl', help='Name of the labelmap file, if different than labelmap.txt',
                    default='/models/image_classification/labelmap.txt')
# Set camera resolution and cpu type
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

# Initialize parameters for object detection
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

# Initialize parameters for image classification
MODEL_NAME_CL = args.modeldir_cl
GRAPH_NAME_CL = args.graph_cl
LABELMAP_NAME_CL = args.labels_cl

min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
c = 0
lastUploaded = datetime.datetime.now()
min_upload_seconds = 3

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter_od from tflite_runtime, else import from regular tensorflow
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
print(CWD_PATH)
# Set paths to object detection files
# Path to .tflite file, which contains the model that is used for object detection (od)
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Set paths to image classification files
# Path to .tflite file, which contains the model that is used for image classification (cl)
PATH_TO_CKPT_CL = os.path.join(CWD_PATH,MODEL_NAME_CL,GRAPH_NAME_CL)

# Path to label map file
PATH_TO_LABELS_CL = os.path.join(CWD_PATH,MODEL_NAME_CL,LABELMAP_NAME_CL)

# Load the label map
with open(PATH_TO_LABELS_CL, 'r') as f:
    labels_cl = [line.strip() for line in f.readlines()]


# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Step 1: Load the Tensorflow Lite model for object detection (od).
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter_od = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter_od = Interpreter(model_path=PATH_TO_CKPT)

interpreter_od.allocate_tensors()

# Get model details for object detection
print("[INFO] loading model...")
input_details = interpreter_od.get_input_details()
output_details = interpreter_od.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
if floating_model:
    print('Detection model is of type float32.')

# Step 2: Load the Tensorflow Lite model for image classification (cl).
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter_cl = Interpreter(model_path=PATH_TO_CKPT_CL,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT_CL)
else:
    interpreter_cl = Interpreter(model_path=PATH_TO_CKPT_CL)

interpreter_cl.allocate_tensors()

# Get model details for image classification
print("[INFO] loading model...")
input_details_cl = interpreter_cl.get_input_details()
output_details_cl = interpreter_cl.get_output_details()
height_cl = input_details_cl[0]['shape'][1]
width_cl = input_details_cl[0]['shape'][2]

floating_model_cl = (input_details_cl[0]['dtype'] == np.float32)
if floating_model_cl:
    print('Classification model is of type float32.')

# will be used for normalization, see line 196
input_mean = 127.5
input_std = 127.5


# Initialize frame rate calculation
# frame_rate_calc = 1
# freq = cv2.getTickFrequency()

# Initialize video stream
print("[INFO] starting video stream...")
videostream = VideoStream(resolution=(imW,imH),framerate=90).start()
time.sleep(1)
fps = FPS().start()

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    # t1 = cv2.getTickCount()
    
    # Set the current timestamp
    timestamp = datetime.datetime.now()

    # Grab frame from video stream
    frame1 = videostream.read()

    # OD: Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # CL: Acquire frame and resize to expected shape [1xHxWx3]
    frame_cl = frame1.copy()
    frame_rgb_cl = cv2.cvtColor(frame_cl, cv2.COLOR_BGR2RGB)
    frame_resized_cl = cv2.resize(frame_rgb_cl, (width_cl, height_cl))
    input_data_cl = np.expand_dims(frame_resized_cl, axis=0)

    # OD: Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # CL: Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model_cl:
        input_data_cl = (np.float32(input_data_cl) - input_mean) / input_std

    # Perform the actual object detection by running the model with the image as input
    interpreter_od.set_tensor(input_details[0]['index'],input_data)
    interpreter_od.invoke()

    # Retrieve detection results
    boxes = interpreter_od.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter_od.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter_od.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter_od.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
            # Set timestamp format for saving
            # ts = timestamp.strftime("%d-%B-%Y_%I:%M:%S%p")
            ts = timestamp.strftime('%Y-%m-%d_%X')  # 2020-03-14_15:32:52

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Gather label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            
            # See if enough time between uploads has passed
            if (timestamp - lastUploaded).seconds >= min_upload_seconds:
                # If label is "bird" or "cat" then upload
                if object_name in("bird","cat"):
                    if object_name == "bird":
                        # Perform the actual image classification by running the model with the image as input
                        interpreter_cl.set_tensor(input_details_cl[0]['index'],input_data_cl)
                        interpreter_cl.invoke()

                        # Retrieve classification result
                        # classes_cl is numpy ndarray (1,num_classes) -> argmax get index of biggest value
                        classes_cl = interpreter_cl.get_tensor(output_details[0]['index']).argmax() # Class index of classified species
                        label_species = labels_cl[classes_cl]

                    # Get label
                    if object_name == "bird":
                        label = '%s: %d%% %s' % (object_name, int(scores[i]*100), label_species) # Example: 'bird: 72% Blackbird'
                    else:
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'cat: 72%'
                    
                    # Draw box and annotate label
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    # Create temporary snapshot
                    t = TempImage()
                    cv2.imwrite(t.path, frame)
                    
                    # Upload temporary file to dropbox and cleanup temporary file
                    #dropbox_path = "/{base_path}/{timestamp}.jpg".format(
                    #    base_path=your_base_path, timestamp=ts)
                    if object_name == "bird":
                        label_species.replace(' ','-')  # e.g. Coal tit -> Coal-tit
                        dropbox_path = f'/{your_base_path}/{ts}_{object_name}_{label_species}.jpg'
                    else:
                        dropbox_path = f'/{your_base_path}/{ts}_{object_name}_.jpg'

                    client.files_upload(open(t.path,"rb").read(),dropbox_path)
                    print("[UPLOADING...] {}".format(ts))
                    t.cleanup()
                    
                    # Trigger IFTTT notification
                    #requests.post('https://maker.ifttt.com/trigger/bird_surveillance/with/key/c2nrke4lrR-FSkYsBS2EWX')
            
                    # Set last Upload to current time
                    lastUploaded=timestamp
            else:
                None

    # Draw framerate in corner of frame
    # cv2.putText(frame,'FPS: {0:.2f}'.format(fps.fps(),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA))

    # All the results have been drawn on the frame, so it's time to display it.
    
    frame_display = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Object detector', frame_display)

    # Calculate framerate
    # t2 = cv2.getTickCount()
    # time1 = (t2-t1)/freq
    # frame_rate_calc= 1/time1
    
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
