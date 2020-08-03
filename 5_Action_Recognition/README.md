# Action Recognition


## About this module

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
Run the file:
`python action_recognition.py`


