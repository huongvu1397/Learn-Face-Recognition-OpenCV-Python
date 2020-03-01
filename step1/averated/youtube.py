import cv2
import numpy as np
from numpy import cov
import math

# anh la hinh vuong W=H=N
# M la so anh = 5
#P la so nguoi torng anh = 1
# 64x64 
# mo ta vector 64x64 chieu
# moi anh la mot diem trong khong gian 4096 chieu

imgHung = cv2.imread("./step1/averated/images/hung.jpg")
imgTien = cv2.imread("./step1/averated/images/tien.jpg")
imgDuy = cv2.imread("./step1/averated/images/duy.jpg")
imgThom = cv2.imread("./step1/averated/images/thom.jpg")
imgThuy = cv2.imread("./step1/averated/images/thuy.jpg")

#cac anh ma tran 3x3
grayHung = cv2.cvtColor(imgHung, cv2.COLOR_BGR2GRAY)
grayTien = cv2.cvtColor(imgTien, cv2.COLOR_BGR2GRAY)
grayThom = cv2.cvtColor(imgThom, cv2.COLOR_BGR2GRAY)
grayThuy = cv2.cvtColor(imgThuy, cv2.COLOR_BGR2GRAY)
grayDuy = cv2.cvtColor(imgDuy, cv2.COLOR_BGR2GRAY)


#ri-> matran[N^2 x 1]
mtr1 = np.reshape(grayHung,-1).reshape(-1,1)
mtr2 = np.reshape(grayTien,-1).reshape(-1,1)
mtr3 = np.reshape(grayThom,-1).reshape(-1,1)
mtr4 = np.reshape(grayThuy,-1).reshape(-1,1)
mtr5 = np.reshape(grayDuy,-1).reshape(-1,1)

meanMatrix = np.array([mtr1,mtr2,mtr3,mtr4,mtr5]).T
print(meanMatrix)




