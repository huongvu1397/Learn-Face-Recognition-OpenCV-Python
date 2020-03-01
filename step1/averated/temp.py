from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
print("mean - gia tri ky vong")
print(M)
# center columns by subtracting column means
C = A - M
print("vector sai số ứng với mỗi matrix")
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print("hiep phuogn sai")
print(V)
print("tri rieng vector rieng")
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)