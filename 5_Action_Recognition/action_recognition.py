'''
Real Time Action Recognition
1. Pose Estimation using OpenPose for locating skeletal joints
2. Tracking by finding the Euclidean distance between the joints of two skeletons
3. Action Recognition using a 3 layered neural network

This file can be run on
(1) a video
(2) a folder of images
(3) or web camera
for testing Real Time Action Recognition
'''

import argparse
import numpy as np
import cv2
if True:  # This is required to include the project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
     ######################################
    # This is to include all the required libraries
    import utils.lib_commons as lib_commons
    import utils.lib_images_io as lib_images_io
    import utils.lib_plot as lib_plot
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import *

def parse(path):
    return ROOT + path if (path and path[0] != "/") else path

def get_cmd_line_arguments():

    def parse_arg():
        parser = argparse.ArgumentParser(
            desc="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='model/trained_classifier.pickle')
        parser.add_argument("-t", "--data_type", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="",
                            help="path to a video file, or images folder, or webcam. \n"
                            "For video and folder, the path should be "
                            "absolute or relative to this project's root. "
                            "For webcam, either input an index or device name. ")
        parser.add_argument("-o", "--output_folder", required=False, default='output/',
                            help="Which folder to save result to.")

        arg = parser.parse_arg()
        return arg
    arg = parse_arg()
    if arg.data_type != "webcam" and arg.data_path and arg.data_path[0] != "/":
        arg.data_path = ROOT + arg.data_path
    return arg

def get_dest_folder_name(source_data_type, source_data_path):
    # This is used to compute a output folder name based on data_type and data_path

    assert(source_data_type in ["video", "folder", "webcam"])

    # /root/data/video.avi --> video
    if source_data_type == "video":  
        folder_name = os.path.basename(source_data_path).split(".")[-2]

    # /root/data/video/ --> video
    elif source_data_type == "folder":  
        folder_name = source_data_path.rstrip("/").split("/")[-1]

    # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
    elif source_data_type == "webcam":
        folder_name = lib_commons.get_time_string()

    return folder_name


arg = get_cmd_line_arguments()

SOURCE_DATA_TYPE = arg.data_type
SOURCE_DATA_PATH = arg.data_path
SOURCE_MODEL_PATH = arg.model_path

DEST_FOLDER_NAME = get_dest_folder_name(SOURCE_DATA_TYPE, SOURCE_DATA_PATH)
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s5_test.py"]

CLASSES = np.array(cfg_all["classes"])
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

# Number of frames used to extract features
WINDOW_SIZE = int(cfg_all["features"]["window_size"])

# Output folder
DEST_FOLDER = arg.output_folder + "/" + DEST_FOLDER_NAME + "/"
DEST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
DEST_VIDEO_NAME = cfg["output"]["video_name"]
DEST_VIDEO_FPS = float(cfg["output"]["video_fps"])

# For webcam, set the max frame rate
SOURCE_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                           ["webcam_max_framerate"])

# For video, set the sampling interval
SOURCE_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                ["video_sample_interval"])

# Openpose settings
OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

# Display settings
img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])

def select_images_loader(source_data_type, source_data_path):
    if source_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            source_data_path,
            sample_interval=SOURCE_VIDEO_SAMPLE_INTERVAL)

    elif source_data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=source_data_path)

    elif source_data_type == "webcam":
        if source_data_path == "":
            webcam_idx = 0
        elif source_data_path.isdigit():
            webcam_idx = int(source_data_path)
        else:
            webcam_idx = source_data_path
        images_loader = lib_images_io.ReadFromWebcam(
            SOURCE_WEBCAM_MAX_FPS, webcam_idx)
    return images_loader

class MultiPersonClassifier(object):
    #For recognizing actions of multiple people

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE, human_id)

    def classify(self, dict_id2skeleton):
        # For classifying the action type of each skeleton in dict_id2skeleton

        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predicting each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)
            print("\n\nPrediction of label for human{}".format(id))
            print("  skeleton: {}".format(skeleton))
            print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        # Getting the classifier based on the person id

        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def remove_skeletons_with_few_joints(skeletons):
    # Removing bad skeletons before sending to the tracker
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            good_skeletons.append(skeleton)
    return good_skeletons


def draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                    skeleton_detector, multiperson_classifier):
    # Drawing skeletons, labels, and prediction scores onto image for display

    # Resizing to a proper size for display
    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
    img_disp = cv2.resize(img_disp,
                          dsize=(desired_cols, img_disp_desired_rows))

    # Drawing all people's skeleton
    skeleton_detector.draw(img_disp, humans)

    # Drawing bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            skeleton[1::2] = skeleton[1::2] / scale_h
            print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            lib_plot.draw_action_result(img_disp, id, skeleton, label)

    # Adding blank to the left for displaying prediction scores of each class
    img_disp = lib_plot.add_white_region_to_left_of_image(img_disp)

    cv2.putText(img_disp, "Frame:" + str(ith_img),
                (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 0), thickness=2)

    # Drawing prediction score for only 1 person
    if len(dict_id2skeleton):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')
        classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp


def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):

    skels_to_save = []
    for human_id in dict_id2skeleton.keys():
        label = dict_id2label[human_id]
        skeleton = dict_id2skeleton[human_id]
        skels_to_save.append([[human_id, label] + skeleton.tolist()])
    return skels_to_save


if __name__ == "__main__":

    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()
    multiperson_classifier = MultiPersonClassifier(SOURCE_MODEL_PATH, CLASSES)
    images_loader = select_images_loader(SOURCE_DATA_TYPE, SOURCE_DATA_PATH)
    img_displayer = lib_images_io.ImageDisplayer()
    os.makedirs(DEST_FOLDER, exist_ok=True)
    os.makedirs(DEST_FOLDER + DEST_SKELETON_FOLDER_NAME, exist_ok=True)
    video_writer = lib_images_io.VideoWriter(
        DEST_FOLDER + DEST_VIDEO_NAME, DEST_VIDEO_FPS)

    try:
        ith_img = -1
        while images_loader.has_image():

            # Reading image
            img = images_loader.read_image()
            ith_img += 1
            img_disp = img.copy()
            print(f"\nProcessing the {ith_img}th image ...")

            # Detecting skeletons
            humans = skeleton_detector.detect(img)
            skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
            skeletons = remove_skeletons_with_few_joints(skeletons)

            # Tracking people
            dict_id2skeleton = multiperson_tracker.track(
                skeletons)  # int id -> np.array() skeleton

            # Recognizing action of each person
            if len(dict_id2skeleton):
                dict_id2label = multiperson_classifier.classify(
                    dict_id2skeleton)

            # Drawing
            img_disp = draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                                       skeleton_detector, multiperson_classifier)

            # Printing label of a person
            if len(dict_id2skeleton):
                min_id = min(dict_id2skeleton.keys())
                print("The prediced label is :", dict_id2label[min_id])

            # Displaying image, and write to video.avi
            img_displayer.display(img_disp, wait_key_ms=1)
            video_writer.write(img_disp)

            # Getting skeleton data and save to file
            skels_to_save = get_the_skeleton_data_to_save_to_disk(
                dict_id2skeleton)
            lib_commons.save_listlist(
                DEST_FOLDER + DEST_SKELETON_FOLDER_NAME +
                SKELETON_FILENAME_FORMAT.format(ith_img),
                skels_to_save)
    finally:
        video_writer.stop()
        print("Action Recognition ends")
 
