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

Dropbox has retired long-lived tokens on September 30th, 2021. There is a guide that offers help. The guide is very long and tiresome. There are two options for tokens that will be used in this guide. 

[More informaiton on the API changes in Dropbox API 2](https://dropbox.tech/developers/migrating-app-permissions-and-access-tokens)

### Fastest way for testing
- Hit the generate token button (The token is valid for about 2-4 hrs) 
- copy token 
- open your directory where the stuff is 
- open file 
- replace existing token example 

### Set up a refresh token
THIS IS SHIT!

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
