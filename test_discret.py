from discretize_map import *
import cv2 as cv

image = cv.imread("image_processing/Astar_test.png")
#cv.imshow('image', image)
segmentation = np.array(image)
#cv.imwrite('color_img.jpg', segmentation[290:330, 680:720, :])
#print(segmentation[300, 700, :])
#cv.imshow("image", segmentation)

#print(segmentation.shape)
#print(segmentation[420, 800, :])
path = discretize_map(segmentation) 