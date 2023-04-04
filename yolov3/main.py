import numpy as np
import cv2
import yolov3CPU

cap = cv2.VideoCapture('D:/OneDrive/OneDrive - Tata Insights and Quants/vision/2_predictions/video.mp4')
ret, frame = cap.read()
height,width,channels = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID') #codec information(XVID) for avi,mp4 etc. extensions
video = cv2.VideoWriter('output.avi', fourcc, 20.0, (width,height))

while(cap.isOpened()):
    #capture frame by frame
    ret, frame = cap.read()
    output = yolov3CPU.objectDetector(frame)
    video.write(output)

# Release the capture
cap.release()
cv2.destroyAllWindows()
video.release()
