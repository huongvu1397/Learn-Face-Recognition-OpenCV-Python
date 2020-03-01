import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

# Add the weighted eigen faces to the mean face     
def createNewFace(*args):
    # Start with the mean image
    output = averageFace

    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        '''
        OpenCV does not allow slider values to be negative. 
        So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
        ''' 
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE/2
        output = np.add(output, eigenFaces[i] * weight)

    # Display Result at 2x size
    output = cv2.resize(output, (0,0), fx=2, fy=2)
    print("255op : ",output)
    cv2.imshow("Result", output)

def resetSliderValues(*args):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
    createNewFace()

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

def createNewFace(*args):
    #Start with the mean image
    output = averageFace
    #Add the eigen faces with the weights
    for i in range(0,NUM_EIGEN_FACES):
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE/2
        output = np.add(output,eigenFaces[i] * weight)
    #Display
    print("output")
    print(len(output))
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
    print(sz)
    #Create data matrix for PCA
    data = createDataMatrix(images)
    print("Data")
    print(data)
    print("Calculating PCA ")
    mean, eigenVectors = cv2.PCACompute(data,mean=None,maxComponents=NUM_EIGEN_FACES)

    covar, mean2 = cv2.calcCovarMatrix(data, 0,cv2.COVAR_SCALE | cv2.COVAR_ROWS | cv2.COVAR_SCRAMBLED)
    print("Mean 1")
    print(mean)
    print("Mean 2")
    print(mean2)
    print("DONE")
    print("Covar ",covar)
    averageFace = mean2.reshape(sz)
    

    eVal, eigenVectors2 = cv2.eigen(covar, True)[1:]

    eigenFaces = []
    for eigenVector in eigenVectors:
            eigenFace = eigenVector.reshape(sz)
            eigenFaces.append(eigenFace)

    # create window for displaying Mean Face
    cv2.namedWindow("Result",cv2.WINDOW_AUTOSIZE)
    output = cv2.resize(averageFace,(0,0),fx=2,fy=2)

    cv2.imshow("Result",output)

    # Create Window for trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    sliderValues = []
	
	# Create Trackbars
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues.append(int(MAX_SLIDER_VALUE/2))
        cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
	
    cv2.imshow("ok",eigenFaces[1])
    cv2.imwrite("./step1/myfig.png",eigenFaces[1])
    plt.imshow(eigenFaces[1],cmap=plt.cm.gray)
    plt.imsave("eigen1.jpg",eigenFaces[0],cmap=plt.cm.gray)
    plt.imsave("eigen2.jpg",eigenFaces[1],cmap=plt.cm.gray)
    plt.imsave("eigen3.jpg",eigenFaces[2],cmap=plt.cm.gray)
    plt.imsave("eigen4.jpg",eigenFaces[3],cmap=plt.cm.gray)
    plt.imsave("eigen5.jpg",eigenFaces[4],cmap=plt.cm.gray)

    plt.show()

	# You can reset the sliders by clicking on the mean image.
    cv2.setMouseCallback("Result", resetSliderValues)




    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    


