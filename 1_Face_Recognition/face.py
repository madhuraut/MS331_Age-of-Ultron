#CODE FOR GAIT Recognition - SIH2020, Team: Age of Ultron, DIAT,Pune
#importing all libraries and models
import os
import cv2
import numpy as np
from scipy import misc
from PIL import Image
import align.detect_face
import pickle
import tensorflow.compat.v1 as tf
from PIL.Image import BILINEAR
import facenet
tf.disable_v2_behavior()


gpuMemoryFraction = 0.3
facenetModelCheckpoint = os.path.dirname(__file__) + "/../model_checkpoints/20180402-114759"
classifierModel = os.path.dirname(__file__) + "/../model_checkpoints/my_classifier_2.pkl"
debug = False

#Face recognition, intially nothing is assigned 
class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None

# Class to identify and add identitiy
class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def identify(self, image):
        faces = self.detect.findFaces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)

        return faces

    def add_identity(self, image, person_name):
        faces = self.detect.findFaces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces


#class for identification
class Identifier:
    def __init__(self):
        with open(classifierModel, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            print(predictions)
            bestClassIndices = np.argmax(predictions, axis=1)
            if predictions[0][bestClassIndices] >= 0.45:  #for more safety 0.5 can be taken

                return self.class_names[bestClassIndices[0]]
            else:
                return 'Unknown'


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            print("Inside face callingg loads module")
            facenet.test_sudhir()
            facenet.load_model(facenetModelCheckpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        imagesPlaceholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phaseTrainPlaceholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhitenFace = facenet.prewhiten(face.image)

        # Running forward pass to calculate embeddings
        feed_dict = {imagesPlaceholder: [prewhitenFace], phaseTrainPlaceholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

# class to define functions for face detection parameters
class Detection:
    minsize = 20  # minimum size of face -- sudhir changed to 5 from 20
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold values -- sudhir changed from [0.6, 0.7, 0.7]
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, faceCropMargin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.faceCropMargin = faceCropMargin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=gpuMemoryFraction)
            sess = tf.Session(config=tf.ConfigProto(gpuOptions=gpuOptions, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def findFaces(self, image):
        faces = []

        boundingBoxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in boundingBoxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.faceCropMargin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.faceCropMargin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.faceCropMargin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.faceCropMargin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)

        return faces
