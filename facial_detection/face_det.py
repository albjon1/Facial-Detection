import cv2
import os
import ctypes

# locating haarcascade in opencv /data folder
casc_path = os.path.dirname(
          cv2.__file__) + '/data/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(casc_path)  # setting as the classifier

try:
    webcam = cv2.VideoCapture(0)  # loading first webcam available index 0

    while True:
        # capturing each frame
        ret, frames = webcam.read()
        grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)  # converting to greyscale
        faces = face_cascade.detectMultiScale(
            grey,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        print(f'Face locations: {faces}')  # location of face(s) in arrays

        # rectangle and text around detected face
        for x1, y1, x2, y2 in faces:             # x-axis, y-axis, width, height
            cv2.putText(frames,
                        'Face Detected',         # text
                        (x1, y1 - 10),           # location
                        cv2.FONT_HERSHEY_PLAIN,  # font
                        1,                       # text size
                        (255, 255, 255))         # BGR 0 - 255

            cv2.rectangle(frames,
                          (x1, y1),              # startpoint
                          (x1 + x2, y1 + y2),    # endpoint
                          (255, 255, 255),       # BGR 0 - 255
                          1)                     # thickness

        # display the resulting frame
        cv2.imshow('Face Detection Application', frames)

        # exit loop if key input equal to ascii 113 ('q')
        key = cv2.waitKey(1)
        if key == 113:
            break
except Exception:
    ctypes.windll.user32.MessageBoxW(
        0, 'An error occurred, try restarting', 'Error Raised', 0)

webcam.release()
cv2.destroyAllWindows()
