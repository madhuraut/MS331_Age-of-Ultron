#Gender Detection
# import libraries and models
import cv2
import os
import numpy as np
from datetime import datetime
from time import sleep
import argparse
from model import WideResNet
from keras.utils.data_utils import get_file


#face recongnition Class
class FaceRecog(object):
    haarPath = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    weightPath = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"


    def __new__(cls, weight_file=None, depth=16, width=8, faceSize=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceRecog, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, faceSize=64):
        self.faceSize = faceSize
        self.model = WideResNet(faceSize, depth=depth, k=width)()
        modelDir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        filePath = get_file('weights.18-4.06.hdf5',
                         self.weightPath,
                         cache_subdir=modelDir)
        self.model.load_weights(filePath)

    def cropFace(self, imgarray, section, margin=40, size=64):
        imgH, imgW, _ = imgarray.shape
        if section is None:
            section = [0, 0, imgW, imgH]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        xa_axis = x - margin
        ya_axis = y - margin
        xa_axis = x + w + margin
        yb_axis = y + h + margin
        if xa_axis < 0:
            xa_axis = min(xa_axis - xa_axis, imgW-1)
            xa_axis = 0
        if ya_axis < 0:
            yb_axis = min(yb_axis - ya_axis, imgH-1)
            ya_axis = 0
        if xa_axis > imgW:
            xa_axis = max(xa_axis - (xa_axis - imgW), 0)
            xa_axis = imgW
        if yb_axis > imgH:
            ya_axis = max(ya_axis - (yb_axis - imgH), 0)
            yb_axis = imgH
        afterCrop = imgarray[ya_axis: yb_axis, xa_axis: xa_axis]
        resizeImg = cv2.resize(afterCrop, (size, size), interpolation=cv2.INTER_AREA)
        resizeImg = np.array(resizeImg)
        return resizeImg, (xa_axis, ya_axis, xa_axis - xa_axis, yb_axis - ya_axis)

    @classmethod
    def drawLbls(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, thickness=3):
        size = cv2.getTextSize(label, font, fontScale, thickness)[0]
        x, y = point
        cv2.putText(image, label, point, font, 1.2, (0,0,255), thickness)

    
    # Out function to detect Gender    
    def detectionOfface(self):
        face_cascade = cv2.CascadeClassifier(self.haarPath)
        #take recorded video as input
        #cap = cv2.VideoCapture('C:/Users/ASUS/Desktop/age/videos/final.mp4')
        #take input from web cam
        cap = cv2.VideoCapture(0)
        def makeVideoSize():
            cap.set(3, 640)
            cap.set(4, 480)
        makeVideoSize()

        while True:
            if not cap.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.faceSize, self.faceSize)
            )
            # Place a placeholder to cropped faces
            faceImgs = np.empty((len(faces), self.faceSize, self.faceSize, 3))
            for i, face in enumerate(faces):
                face_img, afterCrop = self.cropFace(frame, face, margin=40, size=self.faceSize)
                (x, y, w, h) = afterCrop
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                faceImgs[i,:,:,:] = face_img
            if len(faceImgs) > 0:
                # predict ages and genders of the detected faces
                results = self.model.predict(faceImgs)
                predictedGenders = results[0]
                #ages = np.arange(0, 101).reshape(101, 1)
                #predicted_ages = results[1].dot(ages).flatten()
            # draw results
            for i, face in enumerate(faces):
                label = "{}, {}".format(str(datetime.now()),
                                        "Female" if predictedGenders[i][0] > 0.5 else "Male")
                
                self.drawLbls(frame, (face[0], face[1]), label)

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def get_args():
    parseData = argparse.ArgumentParser(description="Detecting faces from web cam, "
                                                 "and estimating gender.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parseData.add_argument("--depth", type=int, default=16,
                        help="N/W depth")
    parseData.add_argument("--width", type=int, default=8,
                        help="N/W width")
    args = parseData.parse_args()
    return args

#Main funtion body
def main():
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceRecog(depth=depth, width=width)

    face.detectionOfface()

#Calling Main Function
if __name__ == "__main__":
    main()
 
