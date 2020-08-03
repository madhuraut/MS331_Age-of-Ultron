#CODE FOR GAIT Recognition - SIH2020, Team: Age of Ultron, DIAT,Pune
#importing all libraries and models
import os
import numpy as np
from scipy.misc import imresize, imread
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import argparse
import pickle
from sklearn.datasets import load_files
from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

##################################
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

# To capture paths to the input images in our dataset
rawData = load_files(os.getcwd() + r'/dataset', shuffle=False)
files = rawData['filenames']
targets = rawData['target']
le=LabelEncoder()
targets = le.fit_transform(targets)
print('Total number of videos are: ', len(files))


print('The categories(levels) are being converted into integers.\nSo, Following is the mapping - \n')
last_labels = zip(range(5), rawData['target_names'])
#################################
# Displaying the first 5 videos (paths) in the training data along with their labels
# (path of video, class label)
for pair in zip(files[:5], targets[:5]):
    print(pair)


# initializing the known encodings and names list
encodings = []
lables = []

poseEstimation = HumanPoseIRNetwork()
gaitRecog = GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 1)

# # Load models, Human3.6m.ckpt,H3.6m-GRU-1 is used for walking actions
poseEstimation.restore('models/Human3.6m.ckpt')
gaitRecog.restore('models/H3.6m-GRU-1.ckpt')
vidNo = 0
for v, leb in zip(files, targets):
    cap = cv2.VideoCapture(v)
    vidNo = vidNo + 1
    print('Processing Video')
    print(vidNo)


    print('Creating Video Frames')

    while(True):
        global ret
        ret = None

        frames = []
        currentFrame = 0


        while(True):

            ret, frame = cap.read()
            if ret == False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imresize(frame, [299, 299])
            batch_imgs = np.expand_dims(frame, 0)
            frames.append(frame)
            # Stop duplicate images
            currentFrame += 1
            if currentFrame == 50:
                break

        if ret == False:
            break
        print(len(frames))
        if len(frames) == 50:
            videoFramesImgs = np.array(frames)
            print(videoFramesImgs.shape)

            print('Video Frames Created')

            # Creating features from input frames in terms of shape (Time, Height, Width, channels)
            spatialFeatures = poseEstimation.feed_forward_features(videoFramesImgs)

            # Process spatial features to generate identification vector
            # Identity Vector will be converted into Linear SVM
            gaitSig, states = gaitRecog.feed_forward(spatialFeatures)
            encodings.append(gaitSig)
            lables.append(leb)

            print(type(gaitSig))
            print(len(gaitSig))
            print('Printing States')
            print(type(states[0][0]))
            print(len(states[0][0]))



model = SVC(kernel='linear', probability=True)
model.fit(encodings, lables)

with open('gaitsvcclassifier7_with_50frames_newrefineddata.pkl', 'wb') as outfile:
    pickle.dump((model, [0, 1, 2, 3, 4]), outfile)
print("Saved classifier model to file gaitsvcclassifier7_with_50frames_newrefineddata.pkl")
 
