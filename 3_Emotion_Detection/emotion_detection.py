'''

################## Emotion DETECTION ########################

'''
###################################################

########## Import Libraries ################

from utils.datasets import labels_fetch
from statistics import mode
import cv2
from utils.inference import faces_detect
from keras.models import load_model
from utils.preprocessor import input_preproc
import numpy as np
from utils.inference import offsets
from utils.inference import model_detection_loading
from utils.inference import bounding_box
from utils.inference import text_show



############ Needed Parameters for loading image and data files. ########################
path_emotion_model = '../trained_models/emotion_models/fer2013_XCEPTION.hdf5'
labels_emotion = labels_fetch('fer2013')
detection_model_path = '../trained_models/face_detection_models/haarcascade_frontalface.xml'


########### Needed Hyper-Parameters for bounding boxes at output screen ###################
labels_emotion = (20, 40)
window_frame = 10


################### Models Need to be loaded ################################
emotion_model = load_model(path_emotion_model, compile=False)
detected_face = model_detection_loading(detection_model_path)
#print("\n Emotion Classifier",emotion_model)


############ Shapes for inference to the input model which we are getting ###################
size_emotion_target = emotion_model.input_shape[1:3]


############### lists of calculated modes ######################
emotion_modes = []


############# Video Stream started here ###################
cv2.namedWindow('Emotion Detection Module')
cap_vid = cv2.VideoCapture('video.mp4')
#'http://192.168.1.107:8080/videos'
while True:
    cap_img = cap_vid.read()[1]
    grayscale_image = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
    color_image = cv2.cvtColor(cap_img, cv2.COLOR_BGR2RGB)
    detected_faces = faces_detect(detected_face, grayscale_image)

    for each_face in detected_faces:

        x1, x2, y1, y2 = offsets(each_face, labels_emotion)
        grayscale_face = grayscale_image[y1:y2, x1:x2]
        try:
            grayscale_face = cv2.resize(grayscale_face, (size_emotion_target))
        except:
            continue

        grayscale_face = input_preproc(grayscale_face, True)
        grayscale_face = np.expand_dims(grayscale_face, 0)
        grayscale_face = np.expand_dims(grayscale_face, -1)
        predicted_emotion = emotion_model.predict(grayscale_face)
        prob_emotion = np.max(predicted_emotion)
        arg_emotion = np.argmax(predicted_emotion)
        text_predicted_emotion = labels_emotion[arg_emotion]
        emotion_modes.append(text_predicted_emotion)

        if len(emotion_modes) > window_frame:
            emotion_modes.pop(0)
        try:
            emotion_mode = mode(emotion_modes)
        except:
            continue

        if text_predicted_emotion == 'angry':
            show_color = prob_emotion * np.asarray((255, 0, 0))
        elif text_predicted_emotion == 'sad':
            show_color = prob_emotion * np.asarray((0, 0, 255))
        elif text_predicted_emotion == 'happy':
            show_color = prob_emotion * np.asarray((255, 255, 0))
        elif text_predicted_emotion == 'surprise':
            show_color = prob_emotion * np.asarray((0, 255, 255))
        else:
            show_color = prob_emotion * np.asarray((0, 255, 0))

        show_color = show_color.astype(int)
        show_color = show_color.tolist()

        bounding_box(each_face, color_image, show_color)
        text_show(each_face, color_image, emotion_mode,
                  show_color, 0, -45, 1, 1)

    cap_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Emotion Detection Module', cap_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap_vid.release()
cv2.destroyAllWindows()
 
