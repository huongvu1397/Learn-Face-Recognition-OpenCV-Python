from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from numpy import linalg as la

#thư mục training
TRAIN_IMG_FOLDER = './step1/imggray/'

train_set_files = os.listdir(TRAIN_IMG_FOLDER)

width  = 100
height = 100

#list ảnh trong thư mục training
print('Lấy list training images...')
train_image_names = os.listdir(TRAIN_IMG_FOLDER)
#Ma trận 5x(100x100) type : float
training_tensor   = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)
print(training_tensor.shape)

#hiển thị ảnh trong list train_image_names
for i in range(len(train_image_names)):
    img = plt.imread(TRAIN_IMG_FOLDER + train_image_names[i])
    training_tensor[i,:] = np.array(img, dtype='float64').flatten()
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
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
#hiển thị ảnh sai khác
for i in range(len(train_image_names)):
    img = normalised_training_tensor[i].reshape(height,width)
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

print("normalised:")
#Ma trận A
a = normalised_training_tensor.T
#Ma trận A.T
b = normalised_training_tensor
AAT = np.mat(a) * np.mat(b)
ATA = np.mat(b) * np.mat(a)

#Tính trị riêng và vector riêng
eigenValues, eigenVectors = la.eig(ATA)

eigenValues = eigenValues.astype(int)

#u = A*v_i
u = []
for i in range(len(eigenVectors)):
    temp =  a* (eigenVectors.T[i].T)
    avI = temp.T
    #u.append(avI)
    # tính giá trị chuẩn hóa của av[i]
    normI = la.norm(avI)
    #print("chuẩn hóa vector u["+str(i)+"]:")
    avv = avI/normI
    print("not norm u["+str(i)+"] = : ",avI)
    print("norm : u = :",avv)
    print("normI : ",normI)
    u.append(avv)

print("u:")
print(u[0])

print("Tính toán cumsum...")
print("sắp xếp theo chiều giảm dần các giá trị..")
sorted_ind = sorted(range(eigenValues.shape[0]), key=lambda k: eigenValues[k], reverse=True)
eigValues_sort = eigenValues[sorted_ind]
eigVectors_sort = eigenVectors[sorted_ind]

print("eig", eigenVectors)
print("eig_sort", eigVectors_sort)

leu = np.cumsum(eigValues_sort)/sum(eigValues_sort)

# Show cumulative proportion of varaince with respect to components M'
print("Cumulative proportion of variance explained vector: \n%s" %leu)
  
#wi =


u_sort = []
for i in range(len(eigVectors_sort)):
    ei = eigVectors_sort.T[i].T
    print("ei : ",ei)
    temp =  a* (ei)
    avI = temp.T
    #u.append(avI)
    # tính giá trị chuẩn hóa của av[i]
    normI = la.norm(avI)
    #print("chuẩn hóa vector u["+str(i)+"]:")
    avv = avI/normI
    print("not norm u["+str(i)+"] = : ",avI)
    u_sort.append(avv)

omg0 = []
omg1 = []
omg2 = []
omg3 = []
omg4 = []
#Chọn M' = 4 
for i in range(4):
    rsubmean = normalised_training_tensor[i]
    wi = (u[i].T) * rsubmean
    omg0.append(wi)

 

print("w0:")
print(omg0)

#plt.imshow(omg0, cmap='gray')
#plt.show()
# End 



print("other")




# calculate covariance matrix
cov_matrix=np.cov(normalised_training_tensor)
cov_matrix = np.divide(cov_matrix,25.0)
print("COV_MATRIX")
print(cov_matrix)
print('Covariance Matrix Shape:', cov_matrix.shape)
#print('Covariance matrix of X: \n%s' %cov_matrix)

#eigenvalues and eigenvectors

eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
print('eigenvalues.shape: {} eigenvectors.shape: {}'.format(eigenvalues.shape, eigenvectors.shape))
print("eigenVectorX")
print(eigenvectors)
print("eigenValuesX")
for i in range(len(eigenvalues)):
    print(eigenvalues[i])

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the eigen pairs in descending order:
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

sorted_ind = sorted(range(eigenvalues.shape[0]), key=lambda k: eigenvalues[k], reverse=True)

eigvalues_sort = eigenvalues[sorted_ind]
eigvectors_sort = eigenvectors[sorted_ind]
train_set_files_sort = np.array(train_set_files)[sorted_ind]

var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

# Show cumulative proportion of varaince with respect to components
print("Cumulative proportion of variance explained vector: \n%s" %var_comp_sum)

# x-axis for number of principal components kept
num_comp = range(1,len(eigvalues_sort)+1)
plt.title('Cum. Prop. Variance Explain and Components Kept')
plt.xlabel('Principal Components')
plt.ylabel('Cum. Prop. Variance Expalined')

plt.scatter(num_comp, var_comp_sum)
plt.show()

reduced_data = np.array(eigvectors_sort[:25]).transpose()
print(reduced_data)
reduced_data.shape

print(training_tensor.transpose().shape, reduced_data.shape)


proj_data = np.dot(training_tensor.transpose(),reduced_data)
proj_data = proj_data.transpose()
proj_data.shape

for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(height,width)
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()
