# Dropbox guide
You can store stuff on your local raspberry, but it does not have much storage & has no iftt integration. So Cloud it is. 
We checked Google Drive and Dropbox and although both are not very user friendly Dropbox is a bit easier.

## Get Dropbox Account
If you don't have one: get it. 

## Create App
- Go to: https://www.dropbox.com/developers/apps
- Hit create app 

<p align="center">
  <img src="/doc/Dropbox_app_create.png">
</p>

## Setup App
- enable add. users
- go to permissions and just enable everything (trust me it works ONLY this way. Tested it for 8hrs +)

<p align="center">
  <img src="/doc/Dropbox_app_permissions.png">
</p>

## Switch to raspberrypi
- The token is godforsaken long. No chance you type it without typos. 
- Therefore: open browser
- Open URL: https://www.dropbox.com/developers/apps
- Select the right app

## Create token
- Hit the generate token button. No clue how this token is valid, dropbox has not documentation on it. 
- copy token 
- open your directory where the stuff is 
- open file 
- replace existing token example 

**Side note:** No, there is currently no option to generate a longer running one. There was change in functionality somewhen in 2021-09.
[This guide](https://www.dropbox.com/developers/documentation/http/documentation#oauth2-authorize) was checked and tested, but NONE of it worked. 
Only thing that worked: hit the generate token button and live with it. 

## Go to terminal 
- initialise env 

```
source tflite1-env/bin/activate
```

- run model  
```
python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model
```

## Please do not forgot to test. Dropbox is not nice. 
