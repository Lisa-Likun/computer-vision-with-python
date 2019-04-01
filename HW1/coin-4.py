import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pylab import *
from matplotlib import pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
img = cv2.imread(dir_path + '/../../IMG_8256.JPG')
# img.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)


img1 = img[500:3000, 500:2500]
roi = img[500:3000, 500:2500]

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 23, 101, 101)
gray = cv2.GaussianBlur(gray, (13, 11), 0)
gray_blur = cv2.medianBlur(gray, 7)

circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 0.9, 120, param1=50, param2=30, minRadius=100, maxRadius=180)
circles_rnd = np.uint16(np.around(circles))
count = 1
for i in circles_rnd[0, :]:
    cv2.circle(roi, (i[0], i[1]), i[2], (0, 0, 150), 15)
    cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 0), 5)
    cv2.putText(roi, str(count), (i[0] - 70, i[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 3.3, (0, 0, 150), 15)
    count += 1

plt.rcParams["figure.figsize"] = (16, 9)
plt.imshow(roi)

#
