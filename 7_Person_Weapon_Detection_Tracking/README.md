# Person & Weapon Detection & Tracking

## About this module

This solution implements YOLOv3 and Deep SORT in order to perform real-time Object Tracking and Weapon Detection.
Yolov3 is an algorithm that uses deep convolutional neural networks to perform Object Detection.
We feed these object detections into Deep SORT in order for a real-time object tracker to be created.

### YOLOv3 Detection
Deep Learning Framework: TensorFlow 
Input: image (frame) 
Number of layers: 106 
Output: bounded box around the detected object, along with the class label

### Deep SORT Tracking
Deep Learning Framework: TensorFlow
Input: image with object(s) detected using YOLOv3
Number of layers: 11 
Output: real time tracking of the detected object






## How to use

`python tracking.py`


