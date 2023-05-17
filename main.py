'''
  Nama       : Eko Rahayu Widodo
  NIM        : 1944190045
  Matkul UTS : Pengolahan Citra
  Project    : Object Color Detection using OpenCV & Python
'''

# import dependencies
import cv2
import numpy as np
from PIL import Image

# set lower and upper to color green
lower_green = np.array([50, 100,100])
upper_green = np.array([70, 255, 255])

# open file video
cap = cv2.VideoCapture("assets/video_testing.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()

    # to change the color space of an image
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # masking
    mask = cv2.inRange(hsvImage, lower_green, upper_green)
    mask_ = Image.fromarray(mask)

    # Sets two vectors to the minimum and maximum corners of the bounding box for the geometry
    bbox = mask_.getbbox()

    # print in the terminal whether the object is detected or not
    print(bbox)

    if bbox is not None:
        # mark the object in red
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
    
    # open frame video
    frameSize = cv2.resize(frame, (340, 540))
    cv2.imshow('frame', frameSize)
    
    # to stop the video using the "q" key on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

# to close the video frame
cap.release()
cv2.destroyAllWindows()