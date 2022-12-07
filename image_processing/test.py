import cv2 as cv
import matplotlib.pyplot as plt

camera_device = 1
cap = cv.VideoCapture(camera_device)
print(cap.isOpened())
success, frame = cap.read()
plt.imshow(frame)
plt.show()