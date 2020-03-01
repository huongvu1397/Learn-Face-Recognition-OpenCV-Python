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

# dummy data

grayHung = np.array([1,2,3,4])
grayTien = np.array([5,6,7,8])
grayThom = np.array([1,3,5,9])
grayThuy = np.array([4,3,2,1])
grayDuy = np.array([1,6,4,2])

#ri -> maTran[N^2 x 1]
mtr1 = np.reshape(grayHung,-1).reshape(-1,1)
mtr2 = np.reshape(grayTien,-1).reshape(-1,1)
mtr3 = np.reshape(grayThom,-1).reshape(-1,1)
mtr4 = np.reshape(grayThuy,-1).reshape(-1,1)
mtr5 = np.reshape(grayDuy,-1).reshape(-1,1)

print("mtr1 :")
print(mtr1)
print("mtr2 :")
print(mtr2)
print("mtr3 :")
print(mtr3)
print("mtr4 :")
print(mtr4)
print("mtr5 :")
print(mtr5)

imgSum = (mtr1 + mtr2 + mtr3 + mtr4 + mtr5)
#tinh toan gia tri trung binh - vector kỳ vọng của toàn bộ dữ liệu
M = imgSum/5
#print("SUM :")
#print(imgSum)
#M = np.divide(imgSum,5)
print("M :")
print(M)
#cv2.imshow("ok",M)
#cv2.waitKey()
#mean = np.mean(imgSum.T,axis = 1)
#print(imgSum)
#print(mean)

#tru di gia tri trung binh
# vector sai số ứng với mỗi ảnh
p1 = mtr1 - M
p2 = mtr2 - M
p3 = mtr3 - M
p4 = mtr4 - M
p5 = mtr5 - M

print("p1 :")
print(p1)
print("p2 :")
print(p2)
print("p3 :")
print(p3)
print("p4 :")
print(p4)
print("p5 :")
print(p5)

print("p1.T :")
print(p1.T)
print("p2.T :")
print(p2.T)
print("p3.T :")
print(p3.T)
print("p4.T :")
print(p4.T)
print("p5.T :")
print(p5.T)


# ma trận hiệp phương sai

V1 = p1*(p1.T)
V2 = p2*(p2.T)
V3 = p3*(p3.T)
V4 = p4*(p4.T)
V5 = p5*(p5.T)

#print("V1 :")
#print(V1)

sqrt5 =math.sqrt(5)
#A = np.divide((p1*p2*p3*p4*p5),sqM)
a = np.array([p1,p2,p3,p4,p5])
A = (1/sqrt5)*(a.T)
print("A :")
print(A)
C = A*(A.T)
print("C :")
print(C)
#Z = np.cov([p1[0,:],p2[1,:],p3[2,:],p4[3,:],p5[4,:]])
#print("Z :")
#print(Z)
#A = C2/()
C2 = (V1+V2+V3+V4+V5)/5
print("C2 :")
print(C2)

# trị riêng Eigenvalues , vector riêng Eigenvectors
#D,V  = np.linalg.eig(C)


#np.linalg.norm(x)

#print(D)

#cv2.imshow("ok",V)
#cv2.waitKey()