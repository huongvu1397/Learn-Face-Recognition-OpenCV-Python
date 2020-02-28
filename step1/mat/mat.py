import cv2
import numpy as np
import dlib
from PIL import Image

# Checking the OpenCV version
print("OpenCV version", cv2.__version__)

# Checking the Numpy version
print("Numpy version", np.__version__)

# Checking the dlib version
print("Dlib version", dlib.__version__)

# Rendering PCA images
def pca(X):
    # Principal Component Analysis
    # input: X, matrix with training data as flattened arrays in rows
    # return: projection matrix (with important dimensions first),
    # variance and mean

    #get dimensions
    num_data,dim = X.shape

    #center data
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X

    if dim>100:
        print('PCA - compact trick used')
        M = dot(X,X.T) #covariance matrix
        e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T #this is the compact trick
        V = tmp[::-1] #reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
    else:
        U,S,V = linalg.svd(X)
        V = V[:num_data] 
    return V,S,mean_X

class PCA:
    pass


