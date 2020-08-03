# Age detection using SSR-Net Model in #-stage CNN
#Import necessary library and modules
import os
import math
import time
from datetime import datetime
from time import sleep
import numpy as np
import cv2 as cv
from age_gender_ssrnet.SSRNET_model import SSR_net_general, SSR_net


#Smaller width and height size will be easier to process
W = 640
H = 480
ot_size = (W, H) #480,340

#Using detecttion model 'haar'
faceDetectorModel = 'haar'

#Using ssrnet for age detection
ageModel = 'ssrnet'


# Initialize numpy random generator
np.random.seed(int(time.time()))

diagnl, ThicknessOfLine = None, None
#Take video input using recorded video
'''
vdo = []
for file_name in os.listdir('videos'):
    file_name = 'videos/' + file_name
    if os.path.isfile(file_name) and file_name.endswith('.mp4'):
        vdo.append(file_name)
srcPath = vdo[np.random.randint(len(vdo))]
#print(srcPath)
cap = cv.VideoCapture(srcPath)
'''
#Capture input stream in real time
cap = cv.VideoCapture(0)

#First need to detect face. So, loading face detection models from particular folders
if (faceDetectorModel == 'haar'):
    faceCascade = cv.CascadeClassifier('face_haar/haarcascade_frontalface_alt.xml')
else:
    faceNet = cv.dnn.readNetFromTensorflow('face_net/opencv_face_detector_uint8.pb', 'face_net/opencv_face_detector.pbtxt')

geNet = None
agNet = None

# Load age and gender models
if (ageModel == 'ssrnet'):
    # Setup global parameters
    facSize = 64
    face_padding_ratio = 0.10
    # Default parameters for SSR-Net
    stgNum = [3, 3, 3]
    lambdaLocal = 1
    lambdaD = 1
    # Initialize gender net
    geNet = SSR_net_general(facSize, stgNum, lambdaLocal, lambdaD)()
    geNet.load_weights('age_gender_ssrnet/ssrnet_gender_3_3_3_64_1.0_1.0.h5')
    # Initialize age net
    agNet = SSR_net(facSize, stgNum, lambdaLocal, lambdaD)()
    agNet.load_weights('age_gender_ssrnet/ssrnet_age_3_3_3_64_1.0_1.0.h5')
else:
    print("Model is not found")

#Function to find the faces from frame. Passing threshold value also
def findFaces(img, confidenceThreshold=0.7):
    #Finds original width and height of images
    H = img.shape[0]
    W = img.shape[1]
    
    faceBox = []

    if (faceDetectorModel == 'haar'):
        #Finding GRAY SCALE of images
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #Detection of faces
        detections = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in detections:
            paddingH = int(math.floor(0.5 + h * face_padding_ratio))
            paddingW = int(math.floor(0.5 + w * face_padding_ratio))
            x1, y1 = max(0, x - paddingW), max(0, y - paddingH)
            x2, y2 = min(x + w + paddingW, W - 1), min(y + h + paddingH, H - 1)
            faceBox.append([x1, y1, x2, y2])
    else:
        #Converting i/p img to 3x300x300 because NN model expects only 300x300 RGB images
        blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), mean=(104, 117, 123), swapRB=True, crop=False)
    
        #Passing blob through model to get detected faces
        faceNet.setInput(blob)
        detections = faceNet.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence < confidenceThreshold):
                continue
            x1 = int(detections[0, 0, i, 3] * W)
            y1 = int(detections[0, 0, i, 4] * H)
            x2 = int(detections[0, 0, i, 5] * W)
            y2 = int(detections[0, 0, i, 6] * H)
            paddingH = int(math.floor(0.5 + (y2 - y1) * face_padding_ratio))
            paddingW = int(math.floor(0.5 + (x2 - x1) * face_padding_ratio))
            x1, y1 = max(0, x1 - paddingW), max(0, y1 - paddingH)
            x2, y2 = min(x2 + paddingW, W - 1), min(y2 + paddingH, H - 1)
            faceBox.append([x1, y1, x2, y2])

    return faceBox

#Function to find the faces from frame. Passing threshold value also
def collectFaces(frame, faceBox):
    faces = []
    # Processing faces
    for i, box in enumerate(faceBox):
        # Converting box coordinates from resized framebground back to original frames
        boxAppr = [
            int(round(box[0] * widthSrc / W)),
            int(round(box[1] * heightSrc / H)),
            int(round(box[2] * widthSrc / W)),
            int(round(box[3] * heightSrc / H)),
        ]
        # Extracting face boxs from original frames
        faceBgrnd = frame[
            max(0, boxAppr[1]):min(boxAppr[3] + 1, heightSrc - 1),
            max(0, boxAppr[0]):min(boxAppr[2] + 1, widthSrc - 1),
            :
        ]
        faces.append(faceBgrnd)
    return faces

#Function to draw boxes calculate height and width and write to o/p video
def calParameters(heightSrc, widthSrc):
    global W, H, diagnl, ThicknessOfLine
    area = W * H
    W = int(math.sqrt(area * widthSrc / heightSrc))
    H = int(math.sqrt(area * heightSrc / widthSrc))
    # Calculating diagonal length
    diagnl = math.sqrt(H * H + W * W)
    # Calculating thickness of lines to draw boxes
    ThicknessOfLine = max(1, int(diagnl / 150))
    # Initialize output video writer
    global out
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('video.avi', fourcc=fourcc, fps=fps, frameSize=(W, H))


#Function to predict ages
def predictAge(faces):
    if (ageModel == 'ssrnet'):
        # Convert faces to N,64,64,3 blob
        blob = np.empty((len(faces), facSize, facSize, 3))
        for i, faceBgrnd in enumerate(faces):
            blob[i, :, :, :] = cv.resize(faceBgrnd, (64, 64))
            blob[i, :, :, :] = cv.normalize(blob[i, :, :, :], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        # Predict gender and age
        Gendrs = geNet.predict(blob)
        ages = agNet.predict(blob)
        #  Construct labels
        labels = ['{}:{}'.format('Age', int(age)) for (gender, age) in zip(Gendrs, ages)]
    else:
        # Convert faces to N,3,227,227 blob
        blob = cv.dnn.blobFromImages(faces, scalefactor=1.0, size=(227, 227),
                                     mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        # Predict gender
        geNet.setInput(blob)
        Gendrs = geNet.forward()
        # Predict age
        agNet.setInput(blob)
        ages = agNet.forward()
        #  Construct labels
        labels = ['{}'.format(Ages[age.argmax()]) for (age) in zip(ages)]
    return labels

#Prosessing in realtime/Video input
paused = False
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculating Parameters height and width
    if (diagnl is None):
        heightSrc, widthSrc = frame.shape[0:2]
        calParameters(heightSrc, widthSrc)
        
    #Converting BGR to HSV and Resizing
    if ((H, W) != frame.shape[0:2]):
        framebground = cv.resize(frame, dsize=(W, H), fx=0, fy=0)
    else:
        framebground = frame
        
    #To Detect faces
    faceBox = findFaces(framebground)
    #Making a copy of original image
    facesBground = framebground.copy()
    if (len(faceBox) > 0):
        # Draw boxes in faces Background ground image
        for (x1, y1, x2, y2) in faceBox:
            cv.rectangle(facesBground, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=ThicknessOfLine, lineType=8)
        
        #Collect all faces into matrix
        faces = collectFaces(frame, faceBox)
    
        #Get age
        labels = predictAge(faces)
        #Getting Timestamp
        cv.putText(facesBground,str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),(140,250), cv.FONT_HERSHEY_SIMPLEX, .9,(0,0,255),2,cv.LINE_AA)
        # Draw labels
        for (label, box) in zip(labels, faceBox):
            cv.putText(facesBground, label, org=(box[0], box[1] - 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=5, lineType=cv.LINE_AA)

    #To show output
    cv.imshow('Faces', facesBground)
    
    # Write output frame
    out.write(facesBground)
    #To quit press on ESC button, pause on SPACE
    key = (cv.waitKey(1 if (not paused) else 0) & 0xFF)
    if (key == 27):
        break
    elif (key == 32):
        paused = (not paused)
    sleep(0.001)
    
cap.release()
cv.destroyAllWindows()
