import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_DIR = './step1/imggray_folder/'
# N = DEFAULT_SIZE = 100
DEFAULT_SIZE = [100, 100] 

# Read images
# lấy hình ảnh khuôn mặt (training faces). 
# Đọc hình ảnh trong thư mục bằng hàm read_images. Hình ảnh đuoặc chia theo các mục con là tên người. 
# Tên thư mục cũng là các mảng riêng để tham chiếu với hình ảnh và tên người.

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

# Assemble Data Matrix
# Biểu diễn hình ảnh dưới dạng vector, vector này có dạng 1x (N*N)
# as_row_matrix = (số ảnh) x (N*N)
# tập hợp các hình ảnh thành một ma trận dữ liệu. Mối hàng của ma trận dữ liệu là một hình ảnh.
def as_row_matrix (X):
    if len (X) == 0:
        return np. array ([])
    mat = np. empty ((0 , X [0].size ), dtype =X [0]. dtype )
    for row in X:
        mat = np.vstack(( mat , np.asarray( row ).reshape(1 , -1))) # 1 x N*N 
    return mat  

# Compute the mean face 
# vector trung bình gồm giá trị kỳ vòng của từng biến 
# ma trận hiệp phương sai bao gồm các phương sai của các biến dọc theo đường chéo chính  và hiệp phương sai giữa mỗi cặp biến.
# ảnh trung bình được tính như sau :
# (ma trận dữ liệu).mean()
[X, y] = read_images()      
average_weight_matrix = np.reshape(as_row_matrix(X).mean( axis =0), X[0].shape)
plt.imshow(average_weight_matrix, cmap=plt.cm.gray)
plt.title("Mean Face")
plt.show()

# Calculate PCA 
# Step1. Trừ mean. Hình ảnh trung bình A phải trừ với mỗi ảnh X
# Step2. Tính toán các vector riêng và trị riêng của ma trận hiệp phương sai S
#      . Ma trận hiệp phương sai của tập hợp m biến là một ma trận vuông (m x m)
#      . Trong đó : Các phần từ nằm trên đường chéo lần lượt là phương sai tương ứng của các biến này.
#                   Các phần tử còn lại là cascp hiệp phương sai của đôi một hai biến ngẫu nhiên khác nhau trong tập hợp.
#      Mỗi eigenvector có cùng chiều (số lượng thành phần) như ảnh gốc, bản thân nó cũng xem như một hình ảnh. 
#      Do đó eigenvector của ma trận hiệp phương sai được gọi là eigenfaces. 
#      Nó là các hướng mà hình ảnh khác với hình ảnh trung bình.
# Step3. Chọn các thành phần chính
#      . Sắp xếp các giá trị riêng với np.argsort theo thứ tự giảm dần và sắp xếp các eigenvector cho phù hợp.
#      . Số lượng thành phần chính K được xác định tùy ý bằng cách đặt ngưỡng trên tổng phương sai. Tổng phương sai X 
#      . n = số lượng hình ảnh, k < n : với công thức get_number_of_components_to_preserve_variance so sánh với 0.95.
# y = W.T (x- mean)  trong đó W = (v1,v2,... vk)




def get_number_of_components_to_preserve_variance(eigenvalues, variance=.95):
    for ii, eigen_value_cumsum in enumerate(np.cumsum(eigenvalues) / np.sum(eigenvalues)):
        print("eigen_value_cumsum : ",eigen_value_cumsum)
        if eigen_value_cumsum > variance:
            return ii+1

def pca (X, y, num_components =0):
    #5x 10000
    [n,d] = X.shape 
    if ( num_components <= 0) or ( num_components >n):
        num_components = n
        mu = X.mean( axis =0).astype(int)
        X = (X - mu ).astype(int)
    if n>d:
        print("n>d")
        C = np.dot(X.T,X).astype(int) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eig(C)
    else :
        print("n<d")
        C = np.dot (X,X.T).astype(int) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eig(C)
        for i in range(len(eigenvectors)):
            print("eigenvectors : ",eigenvectors[i])
        for i in range(len(eigenvalues)):
            print("eigenvalue : ",eigenvalues[i])
        
        eigenvectors = np.dot(X.T, eigenvectors )
        # chuẩn hóa 
        for i in range (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm( eigenvectors [:,i])

    print("ma trận X : ",X)
    print("ma trận C : ",C)

        

    # sort eigenvectors descending by their eigenvalue
    print("Sort--------------------------")
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    print("eigenvector : ",eigenvectors)
    for i in range(len(eigenvalues)):
            print("eigenvalue : ",eigenvalues[i])
    num_components = get_number_of_components_to_preserve_variance(eigenvalues)
    print("num components : ", num_components)
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu]  

[eigenvalues, eigenvectors, mean] = pca (as_row_matrix(X).astype(int), y)

# eigenfaces với giá trị riêng cao nhất được tính bằng tập huấn luyện. 
# Chúng được gọi là ghost faces. ĐƯợc hiển thị bởi các hình dưới.
# Đối với một số dữ liệu ghost faces sắc nét , với một số trường hợp khác, chúng bị mờ đi. Độ sắc nét phụ thuộc vào nền và một số chi tiết của ảnh.


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
        plt.imshow(np.asarray(images[i]) , cmap = colormap )
    if filename is None :
        plt.show()
    else:
        fig.savefig( filename )

        
E = []
number = eigenvectors.shape[1]
for i in range (min(number, 16)):
    e = eigenvectors[:,i].reshape(X[0].shape )
    E.append(np.asarray(e))
# plot them and store the plot to " python_eigenfaces .pdf"
subplot ( title ="Eigenfaces", images=E, rows =4, cols =4, colormap =plt.cm.gray , filename ="python_pca_eigenfaces.png")
plt.show()


# Cumulative sum of first highest eigenvalues is given below. Based on the plot it's clear we should pick these features
# tổng tích lũy của giá trị riêng cao nhất được đưa ra ở đây.

def get_eigen_value_distribution(eigenvectors):
    return np.cumsum(eigenvectors) / np.sum(eigenvectors)

def plot_eigen_value_distribution(eigenvectors, interval):
    plt.scatter(interval, get_eigen_value_distribution(eigenvectors)[interval])

plot_eigen_value_distribution(eigenvalues, range(0, number))
plt.title("Cumulative sum of the first {0} eigenvalues".format(number))
plt.show()

# mỗi khuôn mặt được biểu diễn dưới dạng tổ hợp tuyến tính của các eigenfaces
# mỗi khuôn mặt được ước tính bằng cách sử dụng các eigenfaces tốt nhất, có giá trị riêng lớn nhất và đại diện cho các biến thể lớn nhất trong data.
# Việc reconstruct từ PCA 
# x = Wy + mean với W = (v1,v2,v3,...vk)

#reconstruct 

#Ω_k^T
def project(W , X , mu):
    return np.dot (X - mu , W)
def reconstruct (W , Y , mu) :
    return np.dot (Y , W.T) + mu

#reconstruct ảnh đầu tiên

# [X_small, y_small] = read_images(image_path="./step1/imgdata/") 
# [eigenvalues_small, eigenvectors_small, mean_small] = pca (as_row_matrix(X_small), y_small)

# steps =[i for i in range (eigenvectors_small.shape[1])]
# E = []
# for i in range (len(steps)):
#     numEvs = steps[i]
#     P = project(eigenvectors_small[: ,0: numEvs ], X_small[0].reshape (1 , -1) , mean_small)
#     R = reconstruct(eigenvectors_small[: ,0: numEvs ], P, mean_small)
#     # reshape and append to plots
#     R = R.reshape(X_small[0].shape )
#     E.append(np.asarray(R))
# subplot ( title ="Reconstruction", images =E, rows =4, cols =4, 
#          sptitle ="Eigenvectors ", sptitles =steps , colormap =plt.cm.gray , filename ="python_pca_reconstruction.png")
# plt.show()

# Face Recognition using Eigenfaces
# bây giờ chúng ta sẽ dùng thuật toán để phát hiện khuôn mặt trong hình ảnh unknown image. Trong quá trình nhận dạng, một eigenface được hình thành
# cho hình ảnh khuôn mặt nhất định. Khoảng cách Euclidian giữa eigenface và các eigenfaces đã tìm ra được tính toán. 
# Eigenface với khaorng cách Euclidian nhỏ nhất sẽ giống với người nhất.
#      eigenfaces method sau đó thực hiện nhận dạng bằng cách : 
#           1. Chiếu các ảnh training vào không gian con PCA
#           2. Chiếu hình ảnh nhận dạng vào không gian con PCA
# Đưa ra vector hình ảnh đầu vào, chiếu hình ảnh lên không gian eigenspcaes
# y = W u + u trong đó W = (v1,v2,v3 ,... vk)
#           3. Tìm khoảng cách gần nhất giữa các hình ảnh traing và ảnh nhận dạng.
# // sử dụng khoảng cách Euclide để tính khaorng cách , tuy nhiên có một số thuật toán khác tốt hơn.
#  Khoảng cách Euclidean giữa các điểm p và q là độ dài của đoạn thằng nối chúng 
# d(p,q) = căn bậc 2 của [ ( q1-p1)^2  + (q2 -p1)^2 .... + (qn -pn)^2 ]  
# hay căn bậc 2 của tổng (qi -pi)^2  i -> 1.. n


# p = Ω_k^T train , q = Ω_k^T test

def dist_metric(p,q):
    p = np.asarray(p).flatten()
    q = np.asarray (q).flatten()
    print("p = ",p)
    print("q = ",q)
    print("p-q = ", ((p-q)))
    print("p-q ^2 = ",np. power ((p-q) ,2))
    result = np.sqrt (np.sum (np. power ((p-q) ,2)))
    print("ressult : ",result)
    return result

def predict (W, mu , projections, y, X):
    minDist = float("inf")
    print("minDist = ",minDist)
    minClass = -1
    print("minClass = ",minClass)
    Q = project (W, X.reshape (1 , -1) , mu)
    print("Q : ",Q)
    for i in range (len(projections)):
        dist = dist_metric( projections[i], Q)
        print("dist :",dist)
        if dist < minDist:
            minDist = dist
            minClass = i
    return minClass
#Tính Omk.T


projections = []
# xi - ảnh
for xi in X:
    tempProjections = project (eigenvectors, xi.reshape(1 , -1) , mean)
    print("Ω_k^T : ",tempProjections)
    projections.append(tempProjections)

# ảnh mới
image = Image.open("./step1/imgtest/hung.jpg")
image = image.convert ("L")
if (DEFAULT_SIZE is not None ):
    image = image.resize (DEFAULT_SIZE , Image.ANTIALIAS )
test_image = np. asarray (image , dtype =np. uint8 )
print("test_image ",test_image)
print("test_image ",test_image.reshape(1,-1))
print("mean ",mean)
skImage = (test_image.reshape(1,-1) - mean).reshape(test_image.shape)
print("Φ_i=Γ_i-Ψ",skImage)
plt.imshow(skImage,plt.cm.gray)
#plt.imsave("./temp/img_test_thuy.jpg",skImage,cmap='gray')
plt.show()

# chiếu lên không gian M' chiều
# Φ_i
Z = project (eigenvectors, test_image.reshape (1 , -1) , mean)
print("Z 0 : ",Z)
print("project 0  ",(projections[0])[0])
print("uuuuu ",eigenvectors [:,0])
Xz = -999.0496985 * eigenvectors [:,0].T
print("Xz",Xz)


predicted = predict(eigenvectors, mean , projections, y, test_image)

subplot ( title ="Prediction", images =[test_image, X[predicted]], rows =1, cols =2, 
         sptitles = ["Unknown image", "Prediction :{0}".format(y[predicted])] , colormap =plt.cm.gray , 
         filename ="prediction_test.png", figsize = (5,5))
plt.show()