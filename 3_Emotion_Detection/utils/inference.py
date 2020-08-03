import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

from datetime import datetime
#################

def model_detection_loading(path_of_model):
    detecn_model = cv2.CascadeClassifier(path_of_model)
    return detecn_model
##################################
def faces_detect(detecn_model, image_ary_grey):
    return detecn_model.detectMultiScale(gray_image_ary, 1.3, 5)
####################################################
def bounding_box(face_coords, image_ary, clor):
    x, y, w, h = face_coords
    cv2.rectangle(image_ary, (x, y), (x + w, y + h), color, 2)

def offsets(face_coords, offts):
    x, y, width, height = face_coordinates
    off_x, off_y = offts
    return (x - off_x, x + width + off_x, y - off_y, y + height + off_y)

def text_show(coords, image_ary, text, clor, ofset_x=0, ofset_y=0,
                                                scale_font=2, thicknes=2):
    x, y = coords[:2]
    cv2.putText(image_ary, text, (x + ofset_x, y + ofset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2,color, 3)
   
