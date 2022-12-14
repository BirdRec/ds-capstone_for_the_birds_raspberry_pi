# Define new video class
# Using the legacy camera implementation causes tons of errors, try out the new one
# Author: Friederike Thies after Adrian Rosebrock https://pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
#from pivideostream import PiVideoStream
from threading import Thread
import cv2
import picamera

class PiVideoStream:
    #Camera object that controls video streaming from the Picamera
	def __init__(self, resolution=(1024, 768),framerate=90):
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		# self.stream = PiVideoStream(resolution=resolution,
		# 		framerate=framerate)
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False
	
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	
	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)
			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
	
	def read(self):
		# return the frame most recently read
		return self.frame
	
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
