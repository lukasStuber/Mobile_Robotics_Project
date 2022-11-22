import camera
import cv2
import matplotlib.pyplot as plt
import time

c=camera.Camera()
c.show_webcam()

while(1):
    img=c.take_picture(save=True, file_name="template")
    blur_image=cv2.bilateralFilter(img,20,40,10)
    hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv)
    plt.show()

c.destroy()
