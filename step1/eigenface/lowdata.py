import cv2
import numpy as np
import os

def createDataMatrix(images):
    print("Createing data matrix")
    '''
    Allocate spcae for all images in one data matrix.
    The size of the data matrix is
    (w * h * 3,numImages)
    where,
    w= width image
    h = height image
    3 is 3 color channels
    '''
    numImages = len(images)
    sz = images[0].shape
    print(numImages)
    print(sz)
    data = np.zeros((numImages,sz[0] * sz[1]),dtype=np.float32)
    print(data)
    for i in range(0,numImages):
        image= images[i].flatten()
        data[i,:] = image
    print("DONE data")
    return data


def createNewFace(*args):
        #Start with the mean image
        output = averageFace

        #Add the eigen faces with the weights
        for i in range(0,NUM_EIGEN_FACES):
            weight = MAX_SLIDER_VALUE/2
            output = np.add(output,eigenFaces[i] * weight)
        #Display
        output = cv2.resize(output,(0,0),fx=2,fy=2)
        cv2.imshow("Result",output)
    

if __name__ == '__main__':
    NUM_EIGEN_FACES = 4
    MAX_SLIDER_VALUE = 255
    dirName = "./step1/data/"
    #images = readImages(dirName)
    images =[]
    images.append( np.array([(225,229,48),(251,33,238),(0,255,217)]))
    images.append( np.array([(10,219,24),(255,18,247),(17,255,2)]))
    images.append( np.array([(196,35,234),(232,59,244),(243,57,226)]))
    images.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
    print(images[0])
    print(len(images))
    #Size of images
    sz = images[0].shape
    #Create data matrix for PCA
    data = createDataMatrix(images)
    print("Data")
    print(data)
    print("Calculating PCA ")
    mean, eigenVectors = cv2.PCACompute(data,mean=None,maxComponents=NUM_EIGEN_FACES)
    print("DONE")
    averageFace = mean.reshape(sz)
    A = data - mean
    print((A.T)*A)

    print("Mean")
    print(mean)
    print("eigenVectors")
    print(len(eigenVectors))
    print(eigenVectors[0])
    print(eigenVectors[1])
    print(eigenVectors[2])
    print(eigenVectors[3])

    eigenFaces = []
    for eigenVector in eigenVectors:
            eigenFace = eigenVector.reshape(sz)
            eigenFaces.append(eigenFace)

    # create window for displaying Mean Face
    cv2.namedWindow("Result",cv2.WINDOW_AUTOSIZE)
    output = cv2.resize(averageFace,(0,0),fx=2,fy=2)
    cv2.imshow("Result",output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    


