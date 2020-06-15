import os
import numpy as np
from PIL import Image
from numpy.linalg import inv
import math

#Defind static value
DEFAULT_SIZE = [100, 100]
DEFAULT_IMAGE_DIR = './data/img_train_10/'

class MatrixUtils:

    # Biểu diễn hình ảnh dưới dạng vector, vector này có dạng 1 x (N*N)
    # get_data_matrix = (số ảnh) x (N*N)
    # tập hợp các hình ảnh thành một ma trận dữ liệu. Mối hàng của ma trận dữ liệu là một hình ảnh.
    @staticmethod
    def get_data_matrix (X):
        if len (X) == 0:
          return np. array ([])
        data_matrix = np.empty((0 , X[0].size ), dtype =X [0]. dtype )
        for row in X:
          data_matrix = np.vstack(( data_matrix , np.asarray( row ).reshape(1 , -1))) # 1 x N*N 
        print("tempMatrix : ",data_matrix.shape)
        
        return data_matrix 

    # Read images
    # lấy hình ảnh khuôn mặt (training faces). 
    # Đọc hình ảnh trong thư mục bằng hàm read_images. Hình ảnh đuoặc chia theo các mục con là tên người. 
    # Tên thư mục cũng là các mảng riêng để tham chiếu với hình ảnh và tên người.
    # Input : Thư mục chứa tập huấn luyện
    # Output : mảng matrix của tập ảnh và mảng tên folder chứa ảnh người huấn luyện
    @staticmethod
    def read_images(image_path=DEFAULT_IMAGE_DIR, default_size=DEFAULT_SIZE):
        image_matrixs = []
        face_names = []
        image_dirs = [image for image in os.listdir(image_path) if not image.startswith('.')]
        for image_dir in image_dirs:
            dir_path = os.path.join(image_path, image_dir)
            image_names = [image for image in os.listdir(dir_path) if not image.startswith('.')]
            for image_name in image_names:
                image = Image.open (os.path.join(dir_path, image_name))
                #convert to gray image
                image = image.convert ("L")
                # resize to given size (if given )
                if (default_size is not None ):
                    image = image.resize (default_size , Image.ANTIALIAS )

                image_matrix = np.asarray (image , dtype =np. uint8 )
                image_matrixs.append(image_matrix)
                face_names.append(image_dir)
                #print("images dir: ", image_dir)

        #print("images size: ", len(imageMatrixs))

        return [image_matrixs,image_names]

