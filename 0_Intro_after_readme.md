# Moin from BirdRec

We are Hamburg based students of the NeueFische Data Science Bootcamp and we used the awesome Notebook from [Edje Electronics](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi) as a starting point to set up our own Bird Recognition. The broad steps that you will have to accomplish in order to get this project running are: 

## Get a raspberry pi + camera module

In order to have this working you need: 
- a window compatible bird feeder
- suitable power supply for raspberry
- SD card with at least 2GB
- Camera module with 5MP Megapixel camera or better
- Rasperry pi 3 or newer (older versions might work too though)

## Setup the raspberry pi 

The main steps to set up the pi are: 
- assemble the device, connect and start up using the [raspberry pi documentation](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
- get IP address from your router, [more help](https://www.businessinsider.com/guides/tech/how-to-find-ip-address-of-router)
- If you want to remotely connect: download and setup [VNC viewer](https://www.realvnc.com/en/connect/download/vnc/)
- follow the steps from [here to run the model](1_Raspberry_Pi_Guide.md)

## Setup dropbox 
Setting up dropbox has become more difficult, because there was a change in the API (its now 2.0) - if you have an old access token, this will still work eternally. If not, you will have to follow along with the [Dropbox guide](2_Dropbox_Guide.md)

Tool recommendation for the API calls: [Postman](https://www.postman.com/dropbox-api) because it has a graphic user interface

## Outlook

In the future there is more to come. Right now you will be able to run afterwards a dashboard (see separate notebook) that allows you to build a dashboard that predicts the type of bird from a picture. 

In the future we will add additional guides and options. 