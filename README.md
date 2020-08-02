# DRISHTI

DRISHTI non-invasive biometric system that captures face, expression and gesture of targeted persons (criminals) through a network of distributed  CCTV cameras and also maintains detailed log in a database.

## DRISTI has following sub-modules:
1. Face recognition 
2. Face Anti-spoofing
3. Emotion Recognition
4. Age Detection
5. Gender Detestion
6. Action Recognition
7. Gait Recognition
9. Person-Weapon detection and Tracking
10. Server that aggregates ingormation from all the modules and updates GUI in real-time


### Face recognition 


### Face Anti-spoofing


### Age Detection


### Gender Detestion


### Action Recognition


### Gait Recognition

### Person-Weapon detection and Tracking

### GUI

Whenever an activity is performed by a person, it usually lasts a few seconds. 
It is essential to look into temporal (time) variation of data to predict the activity class with desired accuracy.
Hence, we use OpenPose and a Convolutional Neural Network (CNN).

### Pose Estimation
Deep Learning Model: cmu (pretrained on COCO dataset) 
 Input: image (frame)
 Number of layers: 101
 Output: skeleton is marked on the image (Positions of skeletal joints)

### Tracking
It is used for tracking each person.
Euclidean distance between the joints of two skeletons is used for matching two skeletons.

### Classification
Classification into the various action types is performed with the help of neural network classifier.
 Output: probability of a particular action detected


## How to use

`python action_recognition.py`

