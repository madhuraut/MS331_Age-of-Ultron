# DRISHTI

DRISHTI non-invasive biometric system that captures face, expression and gesture of targeted persons (criminals) through a network of distributed  CCTV cameras and also maintains detailed log in a database.

## DRISTI has following sub-modules:
### 1. Face recognition
![](https://github.com/madhuraut/MS331_Age-of-Ultron/blob/master/demo_vids/face%20recognition.gif)

We are able to identify both known and unknown person in images as well as realtime video stream. 
### 2. Face Anti-spoofing
![](https://github.com/madhuraut/MS331_Age-of-Ultron/blob/master/demo_vids/anti_spoofing.gif)

Most entry level Face recognition systems are susceptible to presentation attaks.
But our system can identify if the face visible in camera frame is reak or fake.
### 3. Emotion Recognition
![](https://github.com/madhuraut/MS331_Age-of-Ultron/blob/master/demo_vids/emotion_recognition.gif)

We are able to detect following emotions from input video feed
[]
### 4. Age/Gender Detestion
![](https://github.com/madhuraut/MS331_Age-of-Ultron/blob/master/demo_vids/gender_age_detection.gif)

Dreishti can dtetct both gender as well as perceivable age of person present in the video frame. We still need to impreove age detection accuray further, we are working on it. 
### 6. Action Recognition
![](https://github.com/madhuraut/MS331_Age-of-Ultron/blob/master/demo_vids/action%20recognition.gif)

Currently we are not looking into temporal information to recognise action being perfoermed. we are using simple neural network to classify the the skeletal data taken from OpenPose. 

We plan to accomodate temporal info by passing OpenPose output to recently opensourced View Adaptive Recurrent Neural Networks.

![](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition/blob/master/README.md)

Actions beigng recognised are:
[Kick, punch, sit, squat, stand, wave, walk, jump, run]
### 7. Gait Recognition
![](https://github.com/madhuraut/MS331_Age-of-Ultron/blob/master/demo_vids/gait_recognition.gif)

We are able to extract gait signature from the skeletal data given by OpenPose and then further classify these gait signatures using SVM classifier. But currently we are heighly overfitting on the training data. We are trying our best to come up with the model that generalises better and thus can be used in real-time.  
### 8. Person-Weapon detection and Tracking
![](https://github.com/madhuraut/MS331_Age-of-Ultron/blob/master/demo_vids/object_face_track.gif)

We are able to detect persons and weapons present in the video ferame using YOLOv3. The bounding box of each person is associated with his/her name using Face Frecognition module. These detections along with thier preper lables from both face Recognition and YOLO are paased to DeepSort Tracking.

So you can see that we can detect weapon present in the video frame and issue warning.

Aslo even if we could capture the face of suspect in any one frame we can track the person no matter howe hard he tries to hide his face. 
### 9. Server that aggregates ingormation from all the modules and updates GUI in real-time

In real time we plan that all these modules will run parallely and send the output to central server which then updates Dashboard GUI in realtime. 

Following is the screen-shot of the proposed GUI:

### Detaiks of each modules are further available in readme files of respective folderrs 


