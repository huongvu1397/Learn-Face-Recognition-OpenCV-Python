from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from numpy import linalg as la

#thư mục training
TRAIN_IMG_FOLDER = './step1/imggray/'
TEST_IMG_FOLDER = './step1/imggray/'

train_set_files = os.listdir(TRAIN_IMG_FOLDER)
test_set_files = os.listdir(TEST_IMG_FOLDER)

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

print('Test Images:')
test_image_names = os.listdir(TEST_IMG_FOLDER)#[i for i in dataset_dir if i not in train_image_names]
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
eigenU = [] 
for i in range(len(eigenVectors)):
    temp =  a* (eigenVectors.T[i].T)
    avI = temp.T
    eigenU.append(avI)
    #u.append(avI)
    print("avI = ",avI)
    # tính giá trị chuẩn hóa của av[i]
    normI = la.norm(avI)
    #print("chuẩn hóa vector u["+str(i)+"]:")
    avv = avI/normI
    u.append(avv)



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




print("OTHER OTHER OTHER OTHER OTHER OTHER OTHER OTHER OTHER")




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

w = np.array([np.dot(proj_data,i) for i in normalised_training_tensor])
print(w.shape)

def recogniser(test_image_names, train_image_names,proj_data,w, t0=2e8, prn=False):

    count        = 0
    num_images   = 0
    correct_pred = 0
    
    result = []
    wts = []
    
    #False match rate (FMR)
    FMR_count = 0
    
    #False non-match rate (FNMR)
    FNMR_count = 0
     

    test_image_names2 = sorted(test_image_names)

    for img in test_image_names2:

        unknown_face = plt.imread(TEST_IMG_FOLDER+img)
        num_images += 1
        
        unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
        normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)

        w_unknown = np.dot(proj_data, normalised_uface_vector)
        diff  = w - w_unknown
        norms = np.linalg.norm(diff, axis=1)
        index = np.argmin(norms)
        
        wts.append([count, norms[index]])

        if prn: print('Input:'+'.'.join(img.split('.')[:2]), end='\t')
        count+=1
        
        match = img.split('_')[0] == train_image_names[index].split('_')[0]
        if norms[index] < t0: # It's a face
            if match:
                if prn: print('Matched:' + train_image_names[index], end = '\t')
                correct_pred += 1
                result.append(1)
            else:
                if prn: print('F/Matched:'+train_image_names[index], end = '\t')
                result.append(0)
                FMR_count += 1
        else:
            if match:
                if prn: print('Unknown face!'+train_image_names[index], end = '\t')
                FNMR_count +=1
                
            else:
                pass
                correct_pred += 1



        if prn: print(norms[index], end=' ')
        if prn: print()
            
            
    
    FMR = FMR_count/num_images
    FNMR = FNMR_count/num_images
    
    
    print('Correct predictions: {}/{} = {} \t\t'.format(correct_pred, num_images, correct_pred/num_images), end=' ')
    print('FMR: {} \t'.format(FMR), end=' ')
    print('FNMR: {} \t'.format(FNMR))
    
    
    
    return wts, result, correct_pred, num_images, FMR, FNMR
    

wts, result, correct_pred, num_images, FMR, FNMR =recogniser(test_image_names, train_image_names,proj_data,w, t0=2e8, prn=True)

def rg(r):
    if r: return 'g'
    else: return 'r'
cl = [rg(r) for r in result]

x=[x[0] for x in wts]
y=[y[1] for y in wts]
plt.scatter(x,y, color=cl, label = 'Distance measure (true ang false pred.)')

x2=[x[0] for x in wts]
y2=[2.7e7 for y in wts]

plt.plot(x2,y2, label = 'Empirical error threshold')
plt.legend()
plt.grid()

plt.show()


CPR_list, t0_list, FMR_list, FNMR_list = [], [] , [] , []
for t0 in np.linspace(start=0, stop=1e8, num=20):
    print('{:e}'.format(t0), end=' ')
    wts, result, correct_pred, num_images, FMR, FNMR = recogniser(test_image_names, train_image_names,proj_data,w, t0)
    
    CPR_list.append(correct_pred/num_images) 
    t0_list.append(t0)
    FMR_list.append(FMR)
    FNMR_list.append(FNMR)

x1=t0_list
y1=FMR_list

x2=t0_list
y2=FNMR_list

x3=t0_list
y3=CPR_list

plt.plot(x1,y1, ls='--', color='r', label='FMR',)
plt.plot(x2,y2, ls='-.', color='b', label='FNMR')
plt.plot(x3,y3, color='g', label='Correct prediction using threshold')

plt.grid()
plt.legend()

count        = 0
num_images   = 0
correct_pred = 0
def Visualization(img, train_image_names,proj_data,w, t0):
    global count,highest_min,num_images,correct_pred
    unknown_face        = plt.imread(TEST_IMG_FOLDER+img)
    num_images          += 1
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
    
    plt.subplot(40,2,1+count)
    plt.imshow(unknown_face, cmap='gray')
    plt.title('Input:'+'.'.join(img.split('.')[:2]))
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    count+=1
    
    w_unknown = np.dot(proj_data, normalised_uface_vector)
    diff  = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)
    
   

    
    plt.subplot(40,2,1+count)
    if norms[index] < t0: # It's a face
            
        match = img.split('_')[0] == train_image_names[index].split('_')[0]
        #if img.split('.')[0] == train_image_names[index].split('.')[0]:
        if match:
            #plt.title('Matched:'+'.'.join(train_image_names[index].split('.')[:2]), color='g')
            plt.title('Matched:', color='g')
            plt.imshow(imread(TRAIN_IMG_FOLDER+train_image_names[index]), cmap='gray')
                
            correct_pred += 1
        else:
            #plt.title('Matched:'+'.'.join(train_image_names[index].split('.')[:2]), color='r')
            plt.title('False matched:', color='r')
            plt.imshow(imread(TRAIN_IMG_FOLDER+train_image_names[index]), cmap='gray')
    else:
        #if img.split('.')[0] not in [i.split('.')[0] for i in train_image_names] and img.split('.')[0] != 'apple':
        if img.split('_')[0] not in [i.split('_')[0] for i in train_image_names]:
            plt.title('Unknown face', color='g')
            correct_pred += 1
        else:
            plt.title('Unknown face', color='r')
                
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    plt.subplots_adjust(right=1.2, top=2.5)
   
    count+=1

    
fig = plt.figure(figsize=(5, 30))

test_image_names2 = sorted(test_image_names)
for i in range(len(test_image_names2)):
    Visualization(test_image_names2[i], train_image_names,proj_data,w, t0=2.7e7)

plt.show()
