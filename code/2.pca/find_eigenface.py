from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from numpy import linalg as la

#thư mục training
TRAIN_IMG_FOLDER = './step1/imggray/'
#thư mục ảnh test
TEST_IMG_FOLDER = './code/convert_gray_image/data/1/'


train_set_files = os.listdir(TRAIN_IMG_FOLDER)
test_set_files = os.listdir(TEST_IMG_FOLDER)

width  = 100
height = 100

#lấy ảnh trong thư mục training
#os.listdir : đọc tất cả các file trong folder nhưng chưa check định dạng
train_image_names = os.listdir(TRAIN_IMG_FOLDER)
# train_image_names = ['gray1.jpg', 'gray2.jpg', 'gray3.jpg', 'gray4.jpg', 'gray5.jpg']
# số lượng ảnh M 
M = len(train_image_names)
#Tạo ma trận M x (width x height) chuyển về định dạng float64   (5 hàng 10000 cột)
training_tensor = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)
print('hiển thị ảnh trong list train_image_names...')
for i in range(len(train_image_names)):
    img = plt.imread(TRAIN_IMG_FOLDER + train_image_names[i])
    training_tensor[i,:] = np.array(img, dtype='float64').flatten()
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

print('hiển thị ảnh trong list test_image_names...')
test_image_names = os.listdir(TEST_IMG_FOLDER)
testing_tensor   = np.ndarray(shape=(len(test_image_names), height*width), dtype=np.float64)

for i in range(len(test_image_names)):
    img = plt.imread(TEST_IMG_FOLDER + test_image_names[i])
    testing_tensor[i,:] = np.array(img, dtype='float64').flatten()
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
    #plt.subplots_adjust(right=1.2, top=1.2)
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

#tính toán mặt trung bình
#khởi tạo ma trận
print("Tìm ảnh trung bình...")
mean_face = np.zeros((1,height*width))
#cộng ảnh
for i in training_tensor:
    print(mean_face)
    mean_face = np.add(mean_face,i)
#chia
mean_face = np.divide(mean_face,float(len(train_image_names))).flatten()
#convert về type int
mean_face = mean_face.astype(int)
#ảnh trung bình
imgMean = mean_face.reshape(height, width)
#hiển thị ảnh trung bình
plt.imshow(mean_face.reshape(height, width), cmap='gray')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

#Tính độ sai khác giữa ảnh huấn luyện Γ_i với ảnh trung bình Ψ : Φ_i=Γ_i-Ψ
print("Tính độ sai khác...")
normalised_training_tensor = np.ndarray(shape=(len(train_image_names), height*width))
for i in range(len(train_image_names)):
    normalised_training_tensor[i] = np.subtract(training_tensor[i],mean_face)
    print(normalised_training_tensor[i].reshape(-1,1))
print("Hiển thị ảnh sai khác...")
for i in range(len(train_image_names)):
    img = normalised_training_tensor[i].reshape(height,width)
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

print("Tìm eigenvector ui của C:")
#Ma trận A (N^2 * M)  (10000,5)
a = normalised_training_tensor.T
#Ma trận A.T (M* N^2) (5,10000)
b = normalised_training_tensor
#A*A.T  = 10000x5 * 5x10000 = 10000 x 10000 => ma trận rất lớn tìm cách khác
AAT = np.mat(a) * np.mat(b)
# A.T*A*vi = ui*vi
# nhân 2 vế với A => (A * A.T* A * v1) = ui*A*vi  
#A.T * A = 5x10000 * 10000x5 = 5x5 => 
ATA = np.mat(b) * np.mat(a)
#Tính trị riêng và vector riêng của ma trận A.T*A (MxM)
print("Tính trị riêng và vector riêng của ma trận A.T *A ...")
eigenValues, eigenVectors = la.eig(ATA)
eigenValues = eigenValues

print("A*v_i là eigenvector của C")
u = []
eigenU = [] 
for i in range(len(eigenVectors)):
    vI = eigenVectors.T[i].T
    calculateAVI =  a* vI
    avI = calculateAVI.T
    eigenU.append(avI)
    #u.append(avI)
    print("avI = ",avI)
    # tính giá trị chuẩn hóa của av[i]
    normI = la.norm(avI)
    #print("chuẩn hóa vector u["+str(i)+"]:")
    avv = avI/normI
    u.append(avv)

print("mảng các vector đặc trưng của C : eigenU = ",eigenU)

print("Tính toán cumsum...")
print("sắp xếp theo chiều giảm dần các giá trị..")
sorted_ind = sorted(range(eigenValues.shape[0]), key=lambda k: eigenValues[k], reverse=True)
eigValues_sort = eigenValues[sorted_ind]
eigVectors_sort = eigenVectors[sorted_ind]

# Show cumulative proportion of varaince with respect to components M'
leu = np.cumsum(eigValues_sort)/sum(eigValues_sort)
print("Cumulative proportion of variance explained vector: \n%s" %leu)
  
#wi =

u_sort = []

for i in range(len(eigVectors_sort)):
    ei = eigVectors_sort.T[i].T
    #print("ei : ",eigVectors_sort.T[i].T)
    temp =  a* (ei)
    avI = temp.T
    #u.append(avI)
    # tính giá trị chuẩn hóa của av[i]
    normI = la.norm(avI)
    #print("chuẩn hóa vector u["+str(i)+"]:")
    avv = avI/normI
    u_sort.append(avv)



omg0 = []
omg1 = []
omg2 = []
omg3 = []
omg4 = []
#Chọn M' = 4 

uNorm = []
eigenU = [] 
for i in range(len(eigenVectors)):
    temp =  a* (eigenVectors.T[i].T)
    avTemp = temp.T
    normTemp = la.norm(avTemp)
    avNormNorm = avTemp/normTemp
    print("calculate norm u["+str(i)+"]:")
    print("temp: ",avTemp)
    print("norm Temp : ",normTemp)
    print("unorm = ",avNormNorm)
    print("avTempshape : ",avTemp.shape)
    eigenU.append (avTemp)
    uNorm.append(avTemp/normTemp)

p0 = normalised_training_tensor[0].reshape(-1,1)
p1 = normalised_training_tensor[1].reshape(-1,1)
p2 = normalised_training_tensor[2].reshape(-1,1)
p3 = normalised_training_tensor[3].reshape(-1,1)
p4 = normalised_training_tensor[4].reshape(-1,1)

#sap xep 
uSort = []
uSort.append(eigenU[2])
uSort.append(eigenU[3])
uSort.append(eigenU[4])
uSort.append(eigenU[1])
uSort.append(eigenU[0])


print("omg 0 : ")
print("w0", uSort[0])
print("w0", p0)
w0 = uSort[0] * p0
w1 = uSort[1] * p0
w2 = uSort[2] * p0
w3 = uSort[3] * p0
print(w0)
print(w1)
print(w2)
print(w3)











print("omg0:",omg0)
print("omg1:",omg1)
print("omg2:",omg2)
print("omg3:",omg3)
print("omg4:",omg4)


#plt.imshow(omg0, cmap='gray')
#plt.show()
# End 



print("tính vector đặc trưng của ảnh")




print("OTHER OTHER OTHER OTHER OTHER OTHER OTHER OTHER OTHER DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
