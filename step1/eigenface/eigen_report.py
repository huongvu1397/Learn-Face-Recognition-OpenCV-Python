from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from numpy import linalg as la


TRAIN_IMG_FOLDER = './step1/imggray/'

train_set_files = os.listdir(TRAIN_IMG_FOLDER)

width  = 100
height = 100

print('Train Images:')
train_image_names = os.listdir(TRAIN_IMG_FOLDER)
training_tensor   = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)

for i in range(len(train_image_names)):
    img = plt.imread(TRAIN_IMG_FOLDER + train_image_names[i])
    training_tensor[i,:] = np.array(img, dtype='float64').flatten()
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
    print("img ["+str(i)+"]")
    print(img)
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

#calculate the mean face

mean_face = np.zeros((1,height*width))

for i in training_tensor:
    print(mean_face)
    mean_face = np.add(mean_face,i)
print("mean_face_1")
print(mean_face)
mean_face = np.divide(mean_face,float(len(train_image_names))).flatten()
print("mean_face_2")
print(mean_face)
mean_face = mean_face.astype(int)
print("int")
imgMean = mean_face.reshape(height, width)
print(imgMean.shape)

plt.imshow(mean_face.reshape(height, width), cmap='gray')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

#Calculation of difference between training vector and mean vector Φ_i=Γ_i-Ψ
normalised_training_tensor = np.ndarray(shape=(len(train_image_names), height*width))

for i in range(len(train_image_names)):
    normalised_training_tensor[i] = np.subtract(training_tensor[i],mean_face)
    print(i)
    print(normalised_training_tensor[i].reshape(-1,1))


for i in range(len(train_image_names)):
    img = normalised_training_tensor[i].reshape(height,width)
    plt.subplot(5,5,1+i)
    plt.imshow(img, cmap='gray')
    plt.imsave("./temp/imgpi_"+str(i)+".jpg",img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

print("normalised:")
a = normalised_training_tensor.T
b = normalised_training_tensor
AAT = np.mat(a) * np.mat(b)
ATA = np.mat(b) * np.mat(a)
print("A=")
print(a)
print("AAT")
print(AAT)
print(AAT.shape)
print("ATA")
print(ATA)
print(ATA.shape)


eigenValues, eigenVectors = la.eig(ATA)

print(len(eigenValues))
print(len(eigenVectors))
print("eigenVector")
print(eigenVectors)
print("eigenValues")
eigenValues = eigenValues.astype(int)
for i in range(len(eigenValues)):
    print(eigenValues[i])

Av = []

for i in range(len(eigenVectors)):
    temp =  a* (eigenVectors.T[i].T)
    Av.append (temp)
    avTemp = temp.T
    print("avTemp["+str(i)+"]")
    print(avTemp)
    print(avTemp.shape)
    normTemp = la.norm(avTemp)
    print("calculate norm u["+str(i)+"]:")
    avv = avTemp/normTemp
    print(avv)
    print(avv.shape)
    

print(eigenValues)
print("cumsum:")
print(np.cumsum(eigenValues))

sorted_ind = sorted(range(eigenValues.shape[0]), key=lambda k: eigenValues[k], reverse=True)
print("sorted:")
print(sorted_ind)

eigvalues_sort = eigenValues[sorted_ind]
print("eigenSort:")
print(eigvalues_sort)
print("sum eigensort")
print(sum(eigvalues_sort))

leu = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

# Show cumulative proportion of varaince with respect to components
print("222222222222 Cumulative proportion of variance explained vector: \n%s" %leu)

print("End Step")  
#end step

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
