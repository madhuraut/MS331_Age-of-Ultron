### Person & Weapon Detection & Face Recognition Tracking
We deetct person and weapons using YOLOv3. Then we corelate detected bonding box with the identity of the perso with the help of FaceNet. Then the detected bonding boxes of person (along with corresponding identity) and weapons is passed to deep sort tracking algorithm.
### About this module

This solution implements YOLOv3 and Deep SORT in order to perform real-time Object Tracking and Weapon Detection.
Yolov3 is an algorithm that uses deep convolutional neural networks to perform Object Detection.
We feed these object detections into Deep SORT in order for a real-time object tracker to be created.

### YOLOv3 Detection
Deep Learning Framework: TensorFlow 
Input: image (frame) 
Number of layers: 106 
Output: bounded box around the detected object, along with the class label
Trained on the dataset: https://cocodataset.org/

### Deep SORT Tracking
Deep Learning Framework: TensorFlow
Input: image with object(s) detected using YOLOv3
Number of layers: 11 
Output: real time tracking of the detected object
Trained on the dataset: Custom dataset created using 2000 "Person" class images and 2000 "Weapon" class images obtained from Open Image Database






### How to use

`python tracking.py`


