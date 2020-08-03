#Code For Weapon Detection Tracking - SIH2020, Team: Age of Ultron, DIAT,Pune
#importing all libraries and models
import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes


#######################
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import face_recognition
import imutils
import pickle
import sklearn
from sklearn import neighbors
from PIL import Image
import edgeiq
import socketio
import base64
import datetime
import MySQLdb

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_of_classes', 80, 'number of classes in the model')

server_address = 'localhost'
stream_fps = 20.0
now = datetime.datetime.utcnow()


total_peep_ctr = None
total_known_peep_ctr = None
total_unknown_peep_ctr = None


sockio = socketio.Client()
#conn = MySQLdb.connect(host="localhost", user="root", password="bokey", db="db_schema")
#now = datetime.datetime.utcnow()

@sockio.event
def connect():
    print('[MESSAGE] Connected to server.')


@scockio.event
def connect_error():
    print('[MESSAGE] Sorry! Could not connect to server.')


@sockio.event
def disconnect():
    print('[MESSAGE] Server Disconnected.')


class ObjctTrakerClient(object):
    def __init__(self, server_addr, stream_fps):
        self.server_addr = server_addr
        self.server_port = 5001
        self._stream_fps = stream_fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)

    def settings(self):
        print('[MESSAGE] Connecting to Server.... http://{}:{}...'.format(
            self.server_addr, self.server_port))
        sockio.connect(
                'http://{}:{}'.format(self.server_addr, self.server_port),
                transports=['websocket'],
                namespaces=['/cv'])
        time.sleep(1)
        return self

    def _coversion_image_to_jpeg_(self, image):
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data_to_server(self, frame, text, unknown, total, known, text_weapon):
        current_t = time.time()
        if current_t - self.last_update_time > self._wait_t:
            self.last_update_time= current_t
            frame = edgeiq.resize(
                    frame, width=640, height=480, keep_scale=True)
            sockio.emit(
                    'cv2server',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(text),
                        'unknown': str(unknown),
                        'known': str(known),
                        'total': str(total),
                        'weapon': '<br />'.join(text_weapon)
                    })

        print('total : ' + str(total))
        print('known : ' + str(known))
        print('unkown : ' + str(unknown))
        print(''.join(text))

    def check_exit(self):
        pass

    def close(self):
        sockio.disconnect()



def main(_argv):
    server_addrss = 'localhost'
    stream_fps_rate = 20.0
    # loading the knn classifier
    print('[Message] Wait, KNN CLASSIFIER IS LOADING...)
    with open('knn_classifire.pickle', 'rb') as f:
        knn_cfier = pickle.load(f)

    # loading the the known faces and embeddings
    print("[Message] Encodings are being loadedloading encodings...")
    data = pickle.loads(open('encodings2.pickle', "rb").read())

    # Parameters defined here
    max_cosine_dist = 0.5
    budget_nn = None
   overlap_nms_max = 1.0

    global ttl_peep_ctr
    global ttl_unknown_peep_ctr
    global ttl_known_peep_cr

    # INitializing deep sort
    file_model = 'model_data/mars-small128.pb'
    encder= gdet.create_box_encoder(model_filename, batch_size=1)
    metrics= nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [cl.strip() for cl in open(FLAGS.classes).readlines()]
    logging.info('The classes are loaded')
    print()
    try:
        streamer=None
        streamer = CVClient(server_addr, stream_fps).setup()
        try:
           
            video = cv2.VideoCapture(0)

        except:
           
            video= cv2.VideoCapture(0)

        out = None

      

        fps = 0.0
        cnt = 0
        old_unknown = 0
        old_length = 0
        old_people_present = []
        while True:
            _, img = video.read()
          

            if img is None:
                logging.warning("The frame is empty")
                time.sleep(0.1)
                count += 1
                if count < 3:
                    continue
                else:
                    break

            img_input0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_input0 = tf.expand_dims(img_in0, 0)
            img_input = transform_images(img_in, FLAGS.size)
            re = img.shape[1] / float(img_in.shape[1])

            time1 = time.time()
            _boxes, _scores, _classes, nums = yolo.predict(img_in)
            _classes = classes[0]
            _name = []
            for i in range(len(classes)):
                _name.append(class_names[int(classes[i])])
            _name = np.array(names)
            convrted_boxes = convert_boxes(img, boxes[0])
            features = encoder(img, converted_boxes)
            _detctions = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(converted_boxes, _scores[0], _name, features)]

            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxes = np.array([d.tlwh for d in detections])
            _scores = np.array([d.confidence for d in detections])
            _classes = np.array([d.class_name for d in detections])
            _indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            _detections = [detections[i] for i in indices]
            total_peep_ctr = 0
            total_known_peep_ctr = 0
            weapons_detected = []

            for i, d in enumerate(_detections):
                if d.class_name == 'person':
                    total_peep_ctr = total_ctr + 1
                    bbox_crop = d.to_tlbr()
                    (bbox_crop[0], bbox_crop[1], bbox_crop[2], bbox_crop[3]) = bbox_crop
                   

                    img_det = img[(bbox_crop[1]).astype(int):(bbox_crop[3]).astype(int),
                             (bbox_crop[0]).astype(int):(bbox_crop[2]).astype(int)]
                   
                    boxes = face_recognition.face_locations(img_de, model='hog')
                  
                    if len(boxes) > 0:
                       face_encodings = face_recognition.face_encodings(img_de, boxes)
                        per_name = knn_clf.predict([encodings_face[0]])
                        
                        closest_dist, closest_index = knn_clf.kneighbors([encodings_face[0]], n_neighbors=1)
                        d.class_name = str(p_name[0])

                        if closest_dist[0][0] < 0.4:
                            name_p = data["names"][closest_index[0][0]]
                            d.class_name = name
                            print(name)
                           

                        else:
                            d.class_name = 'Unknown'
                    else:
                        d.class_name = 'Unknown'
                elif d.class_name == 'knife' or d.class_name == 'scissors' or d.class_name == 'baseball bat' or d.class_name == 'fork':
                    print(d.class_name)
                    detected_weapons.append(d.class_name)
                    d.class_name = 'WEAPON MAY BE POSSIBLEIN THE FRAME'
                    print('DETECTED A WEAPON')
                    cv2.putText(img, 'Warning: weapon detected', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    detections.pop(i)
                    print('removing')
                    print(d.class_name)
                    print(i)
                    print('removed')
          
            
            
            tracker.predict()
            tracker.update(detections)
            
            people_present = []

            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                
                class_name = track.get_class()
                print('from tracker block')
                print(class_name)

                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
                cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)
                if class_name == 'Unknown' or class_name == '' or class_name == None or class_name == []:
                    tracker.tracks.pop(i)
                

            people_present = [t.get_class() for t in tracker.tracks if t.get_class()!="possible weapon" and t.get_class()!= "Unknown"]



            
            old_people_present = people_present
            
            total_known_peep_ctr = len(people_present)


            if total_known_peep_ctr> old_length:
                ayaa(people_present, old_people_present)


            if total_known_peep_ctr < old_length:
                gayaa(old_people_present, old_people_present)
            old_length = total_known_peep_counter


            txt_d = ['person '+i+'\n' for i in people_present]
            weapon_txt = ['Possible Weapon ' + i + '\n' for i in detected_weapons]
           
            fps = (fps + (1. / (time.time() - t1))) / 2
            cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
           
            known = total_known_peep_ctr
            new_unknown = total_peep_ctr - known
           

            if total_peep_ctr== 0:
                known = 0
                added_unknown = 0

            streamer.send_data(img, text, new_unknown, total_peep_counter, known, text_weapon)
            print('Frame Ended')

           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if streamer.check_exit():
                break
    finally:
        if streamer is not None:
            streamer.close()
       
        cv2.destroyAllWindows()

def ayaa(L1, L2):
    print('someone arrived')
    L3 = diff(L1, L2)
    print(L3)

def gayaa(L1, L2):
    print('someone left')
    L3 = diff(L2, L1)
    print(L3)

def diff(L1, L2):
    return list(set(L1) - set(L2))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
