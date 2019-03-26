import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import warnings
import os


cap = cv2.VideoCapture('project_video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    pos_frame = advanced_lane_finding(frame)

    cv2.imshow('frame'pos_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()