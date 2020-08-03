'''
Face Anti-Spoofing
1. Face Detection using OpenCV
2. Face Alignment using landmarks (OpenCV)
3. Face Spoofing Detection using modified ResNet

'''

import argparse
import numpy as np
import imutils
import torch
import models
import cv2
import time
from imutils.video import VideoStream

argp = argparse.ArgumentParser()
argp.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak det")
args = vars(argp.parse_args())

# For Face Detection
protocolPath = "./face_detector/deploy.prototxt"
modPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protocolPath, modPath)

# For Face Alignment
protocolPath2 = "./face_alignment/2_deploy.prototxt"
modPath2 = "./face_alignment/2_solver_iter_800000.caffemodel"
net2 = cv2.dnn.readNetFromCaffe(protocolPath2, modPath2)

# Function to detect face with bounding box
def detect(image):
    frame = imutils.resize(image, width=600)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()

    for i in range(0, det.shape[2]):
        confidence = det[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (400, 400))
            return face
          
          
          
          
#######
# Function to crop aligned face with landmarks
def crop_with_ldmk(image, landmark):
    scale = 3.5
    image_size = 224
    ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
    ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

    std_x, std_y = scale * std_x, scale * std_y

    src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
    dst = np.float32([((image_size - 1) / 2.0, (image_size - 1) / 2.0),
                      ((image_size - 1), (image_size - 1)),
                      ((image_size - 1), (image_size - 1) / 2.0)])
    returnval = cv2.getAffineTransform(src, dst)
    result = cv2.warpAffine(image, returnval, (image_size, image_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return result

# Function to find presentation spoofing attack possibility on the input
def demo(image):
    data= np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    data = data[np.newaxis, :]
    data = torch.FloatTensor(data)
    with torch.no_grad():
        output = model(data)
        output = torch.softmax(output, dim=-1)
        preds = output.to('cpu').numpy()
        attack_prob = preds[:, ATTACK]
    return  attack_prob

if __name__ == '__main__':

    model_name = "MyresNet18"
    load_model_path = "a8.pth"
    model = getattr(models, model_name)().eval()
    model.load(load_model_path)
    model.train(False)

    ATTACK = 1 # For spoofed face
    GENUINE = 0 # For real face
    thresh = 0.90 # Output below this is considered as real face and above it is considered as spoofed face

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    writer = None
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        start = time.time()
        det = net.forward()
        end = time.time()

        print('detect times : %.3f ms'%((end - start)*1000))
        for i in range(0, det.shape[2]):

            confidence = det[0, 0, i, 2]
            if confidence > args["confidence"]:
                box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                sx = startX
                sy = startY
                ex = endX
                ey = endY

                ww = (endX - startX) // 10
                hh = (endY - startY) // 5

                startX = startX - ww
                startY = startY + hh
                endX = endX + ww
                endY = endY + hh

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                x1 = int(startX)
                y1 = int(startY)
                x2 = int(endX)
                y2 = int(endY)

                roi = frame[y1:y2, x1:x2]
                gary_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                resize_mat = np.float32(gary_frame)
                m = np.zeros((40, 40))
                sd = np.zeros((40, 40))
                mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
                new_m = mean[0][0]
                new_sd = std_dev[0][0]
                new_frame = (resize_mat - new_m) / (0.000001 + new_sd)
                blob2 = cv2.dnn.blobFromImage(cv2.resize(new_frame, (40, 40)), 1.0, (40, 40), (0, 0, 0))
                net2.setInput(blob2)
                align = net2.forward()

                aligns = []
                alignss = []
                for i in range(0, 68):
                    align1 = []
                    x = align[0][2 * i] * (x2 - x1) + x1
                    y = align[0][2 * i + 1] * (y2 - y1) + y1
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
                    align1.append(int(x))
                    align1.append(int(y))
                    aligns.append(align1)
                cv2.rectangle(frame, (sx, sy), (ex, ey),(0, 0, 255), 2)
                alignss.append(aligns)

                ldmk = np.asarray(alignss, dtype=np.float32)
                ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]
                img = crop_with_ldmk(frame,ldmk)

                time1 = time.time()
                attack_prob = demo(img)
                time2 = time.time()
                print('prob times : %.3f ms'%((time2-time1)*1000))
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print(attack_prob)

                true_prob = 1 - attack_prob
                if attack_prob > thresh:
                    label = 'FAKE'
                    cv2.putText(frame, label, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 3)
                else:
                    label = 'TRUE'
                    cv2.putText(frame, label, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


        cv2.imshow("Frame", frame)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output3.avi", fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
 
