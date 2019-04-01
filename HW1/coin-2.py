
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pylab import*
from matplotlib import pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)
img=cv2.imread(dir_path+'/../../IMG_8254.JPG')
    #img.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
   # cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)


img1=img[500:3000,500:2500]
roi=img[500:3000,500:2500]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
# print("threshold value %s" % ret)
# cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
# cv2.imshow("binary0", binary)

gray_blur = cv2.medianBlur(gray, 25)
thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,1)
#    #
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations=3)

   #
cont_img = closing.copy()


circles = cv2.HoughCircles(closing, cv2.HOUGH_GRADIENT, 0.2, 120, param1 = 70, param2 = 30, minRadius = 100, maxRadius = 180)
circles_rnd = np.uint16(np.around(circles))

print(circles_rnd.shape)
count = 1
for i in circles_rnd[0, :]:
    cv2.circle(roi, (i[0], i[1]), i[2], (0, 0, 150), 15)
    cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 0), 5)
    cv2.putText(roi, str(count), (i[0] - 70, i[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 3.3, (0, 0, 150), 15)
    count += 1
    plt.rcParams["figure.figsize"] = (16, 9)
plt.figure()
plt.imshow(roi)
plt.show()
# roi,contours,hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area <5000 or area > 105000:
#         continue
#    #
#     if len(cnt) < 5:
#         continue
#    #
#     ellipse = cv2.fitEllipse(cnt)
#     img2=cv2.ellipse(img1, ellipse, (0,255,0), 20)
    #cv2.imshow("Morphological Closing", closing)
    #cv2.imshow("Adaptive Thresholding", thresh)
#     #cv2.imshow('Contours', roi)
#    #
# # if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
#
# # img.release()
# # cv2.destroyAllWindows()
#
# # if __name__ == "__main__":
# #     run_main()
#
#
# # #
# plt.figure()
# plt.imshow(cont_img)
# plt.show()
# print(cont_img)

