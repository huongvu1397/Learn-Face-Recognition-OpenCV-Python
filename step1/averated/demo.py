import numpy as np
from numpy import linalg as la
import cv2
import math


w, v = la.eig(np.array([[5,2],[9,2]]))

# Danh sách trị riêng
print("Eigenvalues: ")
print(w)
# Danh sách vector riêng
print("Eigenvectors: ")
print(v)

a = la.norm(v)

#print(a)


a = [1,2,3]

b = [4,5,6]

#c = a+b

mtr1 = np.reshape(a,-1).reshape(-1,1)
mtr2 = np.reshape(b,-1).reshape(-1,1)

m = mtr1+ mtr2 

print(mtr1)
print(mtr2)

imgAve = np.divide(m,2)

print(m)

#C = np.reshape(c,6).reshape(-1,1)


#21/6 = 3.5
print(np.mean(m))
#
mean2 = np.mean(m,axis = 1,keepdims=True)
print(mean2)
print(imgAve)

