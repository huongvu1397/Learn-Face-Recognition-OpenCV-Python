import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


IMAGE_DIR = './step1/imgdata/'
DEFAULT_SIZE = [100, 100] 

def read_images(image_path=IMAGE_DIR, default_size=DEFAULT_SIZE):
    images = []
    images_names = []
    image_dirs = [image for image in os.listdir(image_path) if not image.startswith('.')]
    for image_dir in image_dirs:
        dir_path = os.path.join(image_path, image_dir)
        image_names = [image for image in os.listdir(dir_path) if not image.startswith('.')]
        for image_name in image_names:
            image = Image.open (os.path.join(dir_path, image_name))
            image = image.convert ("L")
            # resize to given size (if given )
            if (default_size is not None ):
                image = image.resize (default_size , Image.ANTIALIAS )
            images.append(np.asarray (image , dtype =np. uint8 ))
            images_names.append(image_dir)
    return [images,images_names]

def readImagesOpenCV(path):
        print("Reading images from "+ path)
        images= []
        for filePath in sorted(os.listdir(path)):
            fileExt = os.path.splitext(filePath)[1]
            print(filePath)
            if fileExt in [".jpg",".jpeg"]:
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

def as_row_matrix (X):
    if len (X) == 0:
        return np. array ([])
    mat = np. empty ((0 , X [0].size ), dtype =X [0]. dtype )
    for row in X:
        mat = np.vstack(( mat , np.asarray( row ).reshape(1 , -1))) # 1 x r*c 
    return mat

def get_number_of_components_to_preserve_variance(eigenvalues, variance=.95):
    for ii, eigen_value_cumsum in enumerate(np.cumsum(eigenvalues) / np.sum(eigenvalues)):
        if eigen_value_cumsum > variance:
            return ii

def pca (X, y, num_components =0):
    [n,d] = X.shape
    if ( num_components <= 0) or ( num_components >n):
        num_components = n
        mu = X.mean( axis =0)
        X = X - mu
    if n>d:
        C = np.dot(X.T,X) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(C)
    else :
        C = np.dot (X,X.T) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors )
        for i in range (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm( eigenvectors [:,i])
    print("hihi")
    print(len(eigenvalues))
    nonSort = eigenvectors
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    num_components = get_number_of_components_to_preserve_variance(eigenvalues)
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu,nonSort]  

def subplot ( title , images , rows , cols , sptitle ="", sptitles =[] , colormap = plt.cm.gray, filename = None, figsize = (10, 10) ):
    fig = plt.figure(figsize = figsize)
    # main title
    fig.text (.5 , .95 , title , horizontalalignment ="center")
    for i in range ( len ( images )):
        ax0 = fig.add_subplot( rows , cols ,( i +1))
        plt.setp ( ax0.get_xticklabels() , visible = False )
        plt.setp ( ax0.get_yticklabels() , visible = False )
        if len ( sptitles ) == len ( images ):
            plt.title("%s #%s" % ( sptitle , str ( sptitles [i ]) )  )
        else:
            plt.title("%s #%d" % ( sptitle , (i +1) )  )
        im = np.asarray(images[i])
        print(len(im))
        #plt.write("./step1/eigens"+str(i)+".jpg")
        #plt.imsave("./step1/eigens"+str(i)+".jpg",im,cmap=colormap)

        plt.imshow(im , cmap = colormap)
        
        
    if filename is None :
        plt.show()
    else:
        fig.savefig( filename )

def get_eigen_value_distribution(eigenvectors):
    return np.cumsum(eigenvectors) / np.sum(eigenvectors)

def plot_eigen_value_distribution(eigenvectors, interval):
    plt.scatter(interval, get_eigen_value_distribution(eigenvectors)[interval])


if __name__ == '__main__':
    [X, y] = read_images()
     
    average_weight_matrix = np.reshape(as_row_matrix(X).mean( axis =0), X[0].shape)
    plt.imshow(average_weight_matrix, cmap=plt.cm.gray)
    plt.imsave("./step1/train_mean.jpg",average_weight_matrix,cmap=plt.cm.gray)
    plt.title("Mean Face")

    [eigenvalues, eigenvectors, mean,nonSort ] = pca(as_row_matrix(X), y)
    E = []
    number = eigenvectors.shape[1]
    for i in range (min(number, 16)):
        e = eigenvectors[:,i].reshape(X[0].shape )
        E.append(np.asarray(e))

    # plot them and store the plot to " python_eigenfaces .pdf"
    subplot ( title ="Eigenfaces", images=E, rows =4, cols =4, colormap =plt.cm.gray , filename ="./step1/python_pca_eigenfaces.png")


    

    #plot_eigen_value_distribution(eigenvalues, range(0, number))
    #plt.title("Cumulative sum of the first {0} eigenvalues".format(number))
    cv2.waitKey()
    cv2.destroyAllWindows()
    plt.show()

