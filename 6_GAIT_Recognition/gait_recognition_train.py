
#CODE FOR GAIT Recognition - SIH2020, Team Age of Ultron, DIAT,Pune
#importing all libraries and models
import os
import numpy as np
from scipy.misc import imresize, imread
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import argparse
from sklearn.datasets import load_files
from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork
import pickle
from sklearn.svm import SVC

#List of all persons
P_names = ['Aishwaraya', 'Falguni', 'Lucy', 'Madhu', 'Nanda']
font = cv2.FONT_HERSHEY_SIMPLEX
#List of 16 body joints
jointNames = [
    'left wrist',
    'left shoulder',
    'left elbow',
    'left hip',
    'left knee',
    'left ankle',
    'right ankle ',
    'right knee ',
    'right hip',
    'right wrist',
    'right elbow',
    'right shoulder',
    'pelvis',
    'thorax',
    'upper neck',
    'head top'   
]

# Loading Classifier
#print('[INFO] loading classifier')
with open('gaitsvcclassifier3_with_10frames_newrefineddata.pkl', 'rb') as infile:
    (model, class_names) = pickle.load(infile)

# initialization of known encodings and names
encodings = []
lables = []
#for pose detection
net_pose = HumanPoseIRNetwork()
# for GAIT recognition
net_gait = GaitNetwork(recurrent_unit='GRU', rnn_layers=1)
# Load models, Human3.6m.ckpt,H3.6m-GRU-1 is used for walking actions
net_pose.restore('models/Human3.6m.ckpt')
net_gait.restore('models/H3.6m-GRU-1.ckpt')
cap = cv2.VideoCapture('testNanda.mp4')
print('Creating Video Frames')
writer = None


while (True):
    global predictedNames
    global ret
    global frame
    frame = None
    predictedNames = ''
    ret = None
    frames = []
    currentFrame = 0
    while (True):
        ret, frame = cap.read()
        # print(ret)
        if ret == False:
            break

        frame = imresize(frame, [299, 299])
        batch_imgs = np.expand_dims(frame, 0)
        frames.append(frame)


        # Stop duplicate images
        currentFrame += 1
        if currentFrame == 10:
            break
    if ret == False:
        break
    if len(frames) == 10:
        videoFramesImgs = np.array(frames)
        print(videoFramesImgs.shape)
        print('vedio frames created')

        # Creating features from input frames in terms of shape (Time, Height, Width, channels)
        spatial_features = net_pose.feed_forward_features(videoFramesImgs)

        # Process spatial features and generate identification vector
        gait_sig, states = net_gait.feed_forward(spatial_features)
        predictions = model.predict_proba([gait_sig])
        bestClassIndices = np.argmax(predictions, axis=1)
        bestClassProbability = predictions[np.arange(len(bestClassIndices)), bestClassIndices]
        predictedNames = str(P_names[bestClassIndices[0]])
        if bestClassProbability > 0.99:
            text = 'Giat recognised: ' + predictedNames
            cv2.putText(frame, text, (10, 250), font, 0.6, (255, 255, 255), 2)
        cv2.imshow('gait', frame)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output_gait_falguni.avi", fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        # if the writer is not None, write the frame with recognized

        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1) & 0xFF
        #To break execution program of type q
        if key == ord("q"):
            break


        print('Prediction is')
        print(type(bestClassIndices))
        print(bestClassIndices)
        print('with probability of')
        print(bestClassProbability)
    else:
        print('list doest not contain 50 frames.So, may be the last set of 50 frames is short.')


cv2.destroyAllWindows()