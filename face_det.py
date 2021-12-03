import cv2
import os

# locating haarcascade in opencv /data folder
cascPath = os.path.dirname(
          cv2.__file__) + "/data/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)  # setting as the classifier

video_capture = cv2.VideoCapture(0)  # loading first webcam available index 0
while True:
    # capturing each frame
    ret, frames = video_capture.read()
    grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) # converting to greyscale
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # rectangle and text around detected face
    for x1, y1, x2, y2 in faces:  # bottom, left, top, right
        cv2.putText(frames, 'Face Detected', (
            x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        cv2.rectangle(frames, (x1, y1), (x1 + x2, y1 + y2), (255, 255, 255), 1)

    # display the resulting frame
    cv2.imshow('Face Detection Application', frames)

    # exit loop if key equal to ascii 113 ('q')
    key = cv2.waitKey(1)
    if key == 113:
        break

video_capture.release()
cv2.destroyAllWindows()
