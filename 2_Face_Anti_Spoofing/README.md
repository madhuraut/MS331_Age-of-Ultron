# Face Anti-Spoofing
To tackle with presentation attacks on face recognition system it is important to include face anti-spoffing.

## About this module

In order to make face recognition systems more secure, we need to be able to detect such fake/non-real faces. Liveness detection or anti-spoofing refers to such algorithms.
The solution utilizes:
Facial landmark detection for analysis.
Modified ResNet for anti spoofing


### Face Detection
Deep Learning Framework: CAFFE
Input: 300X300px image, 3 color channels 
Number of layers: 112 
Output: confidence value for face detection
Dataset used: https://ibug.doc.ic.ac.uk/resources/300-W/

### Face Alignment
Deep Learning Framework: CAFFE
Input: 40X40px image 
Number of layers: 33
Output: vector of 68 landmarks obtained over the face
Dataset used: https://ibug.doc.ic.ac.uk/resources/300-W/

### Face Spoofing Detection
CNN model: RESNET
Input: Face Region of Interest (ROI)
Number of layers: 23 
Output: spoofing attack probability ranging from 0 to 1
Dataset used: http://chalearnlap.cvc.uab.es/dataset/33/description/




## How to use

`python anti_spoofing.py`



