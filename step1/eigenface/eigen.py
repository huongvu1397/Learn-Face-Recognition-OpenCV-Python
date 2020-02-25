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
    data = np.zeros((numImages,sz[0] * sz[1] * sz[2]),dtype=np.float32)
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
                im = cv2.imread(imagePath)

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
    NUM_EIGEN_FACES = 5
    MAX_SLIDER_VALUE = 255
    dirName = "./step1/data/"
    images = readImages(dirName)
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

    #eigenFaces = []

    #for eigenVector in eigenVectors:
    #        eigenFace = eigenVector.reshape(sz)
    #        eigenFaces.append(eigenFace)
    
    # create window for displaying Mean Face
    cv2.namedWindow("Result",cv2.WINDOW_AUTOSIZE)
    output = cv2.resize(averageFace,(0,0),fx=2,fy=2)
    cv2.imshow("Result",output)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    

