from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import svm
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
import cv2
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
    data = np.zeros((numImages,sz[0] * sz[1]),dtype=np.float32)
    print(data)
    print(numImages)
    for i in range(0,numImages):
        image= images[i].flatten()
        print("Image")
        print(image)
        data[i,:] = image
    print("DONE data")
    print(data)
    return data

def readImages(path):
        print("Reading images from "+ path)
        images= []
        for filePath in sorted(os.listdir(path)):
            fileExt = os.path.splitext(filePath)[1]
            print(filePath)
            if fileExt in [".jpg",".jpeg"]:
                #
                imagePath = os.path.join(path,filePath)
                print(imagePath)
                iz = cv2.imread(imagePath)
                im = cv2.cvtColor(iz, cv2.COLOR_BGR2GRAY)

                if im is None: 
                    print("image:{} no read properly".format(imagePath))
                else :
                    #convert image to floating point 
                    im = np.float32(im)/255.0
                    #add image to list
                    images.append(im)
                    # flip image
                    imFlip = cv2.flip(im,1)
                    # append flipped image
                    images.append(imFlip)
        numImages = int(len(images)/2)
        #Exit if no image found:
        if numImages==0 :
                print("No images found")
                sys.exit(0)
        print(str(numImages)+" files read.")
        return images



plt.show()

if __name__ == '__main__':
    dirName = "./step1/data/"
    images = readImages(dirName)
    print(len(images))
    #Size of images
    sz = images[0].shape
    print(sz)
    #Create data matrix for PCA
    data = createDataMatrix(images)
    print("Data")
    print(data)
    faces = datasets.fetch_olivetti_faces()
    print("faces ",len(faces))
    faces.data.shape

    print(faces.target)

    cl1 = cv2.imread("./step1/data/1.jpg")
    cl2 = cv2.imread("./step1/data/2.jpg")
    cl3 = cv2.imread("./step1/data/3.jpg")
    cl4 = cv2.imread("./step1/data/4.jpg")
    cl5 = cv2.imread("./step1/data/5.jpg")

    img1 = cv2.cvtColor(cl1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cl2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(cl3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.cvtColor(cl4, cv2.COLOR_BGR2GRAY)
    img5 = cv2.cvtColor(cl5, cv2.COLOR_BGR2GRAY)

    #show image
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(3, 5, 1, xticks=[], yticks=[])
    ax.imshow(img1, cmap=plt.cm.bone)
    ax = fig.add_subplot(3, 5, 2, xticks=[], yticks=[])
    ax.imshow(img2, cmap=plt.cm.bone)
    ax = fig.add_subplot(3, 5, 3, xticks=[], yticks=[])
    ax.imshow(img3, cmap=plt.cm.bone)
    ax = fig.add_subplot(3, 5, 4, xticks=[], yticks=[])
    ax.imshow(img4, cmap=plt.cm.bone)
    ax = fig.add_subplot(3, 5, 5, xticks=[], yticks=[])
    ax.imshow(img5, cmap=plt.cm.bone)

    X_train, X_test, y_train, y_test = train_test_split(faces.data,
        faces.target, random_state=0)

    print(X_train.shape, X_test.shape)
