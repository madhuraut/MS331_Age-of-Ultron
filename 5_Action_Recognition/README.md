## Action Recognition


### About this module
OpenPose gives the information about skeletal data from a video frame. We use this skeletal data to train a simple 3- layered neural network. We are able to correctly classify the input video frame into one of the following nine classes
[Kick, punch, sit, squat, stand, wave, walk, jump, run].

Hence, we use OpenPose and a Convolutional Neural Network (CNN).

### Pose Estimation
Deep Learning Model: cmu (pretrained on COCO dataset) 
 Input: image (frame)
 Number of layers: 101
 Output: skeleton is marked on the image (Positions of skeletal joints)
Trained on the dataset: https://cocodataset.org/#home


### Tracking
It is used for tracking each person.
Euclidean distance between the joints of two skeletons is used for matching two skeletons.

### Classification
Classification into the various action types is performed with the help of neural network classifier.
 Output: probability of a particular action detected
Trained on the dataset: https://drive.google.com/open?id=1V8rQ5QR5q5zn1NHJhhf-6xIeDdXVtYs9


### How to use
Run the file:
`python action_recognition.py`


