#Code For Face Recognition - SIH2020, Team: Age of Ultron, DIAT,Pune
#importing all libraries and models
import sys
import time
import cv2
import argparse
import face
import facenet



facenet.testSudhir()

#Code to draw rectangle in face and to put recognized data
def addOverlays(frame, faces, frameRate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frameRate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def parseArguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Debug outputs are enabled.')
    return parser.parse_args(argv)

def main(args):
    # after Number of frames = 3 run face detection
    frameInterval = 3  
    # fps display interval in seconds 
    fpsDisplayInterval = 5 
    #initially frame rate is 0 
    frameRate = 0
    #initially frame count is 0
    frameCount = 0

    cap = cv2.VideoCapture(0)
    startTime = time.time()
    faceRecognition = face.Recognition()
    
    writer = None
    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capturing frame-by-frame
        ret, frame = cap.read()

        if (frameCount % frameInterval) == 0:
            faces = faceRecognition.identify(frame)

            # To check current fps
            endTime = time.time()
            if (endTime - startTime) > fpsDisplayInterval:
                frameRate = int(frameCount / (endTime - startTime))
                startTime = time.time()
                frameCount = 0

        addOverlays(frame, faces, frameRate)

        frameCount += 1
        #To show video frame as output
        cv2.imshow('Video', frame)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output.avi", fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
            
        # if write is not performed, then write the frame with recognized
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
