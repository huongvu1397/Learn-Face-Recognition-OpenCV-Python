from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from numpy import linalg as la
import cv2


TRAIN_IMG_FOLDER = './step1/imggray/'

train_set_files = os.listdir(TRAIN_IMG_FOLDER)

width  = 3
height = 3

print('Train Images:')
train_image_names = os.listdir(TRAIN_IMG_FOLDER)

train_image_names =[]
train_image_names.append( np.array([(225,229,48),(251,33,238),(0,255,217)]))
train_image_names.append( np.array([(10,219,24),(255,18,247),(17,255,2)]))
train_image_names.append( np.array([(196,35,234),(232,59,244),(243,57,226)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(225,229,48),(251,33,238),(0,255,217)]))
train_image_names.append( np.array([(10,219,24),(255,18,247),(17,255,2)]))
train_image_names.append( np.array([(196,35,234),(232,59,244),(243,57,226)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(225,229,48),(251,33,238),(0,255,217)]))
train_image_names.append( np.array([(10,219,24),(255,18,247),(17,255,2)]))
train_image_names.append( np.array([(196,35,234),(232,59,244),(243,57,226)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))
train_image_names.append( np.array([(255,223,224),(255,0,255),(249,255,235)]))

training_tensor   = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)
print("AAAA: training_tensor : ",training_tensor.shape)  

for i in range(len(train_image_names)):
    #img = plt.imread(TRAIN_IMG_FOLDER + train_image_names[i])
    training_tensor[i,:] = np.array(train_image_names[i], dtype='float64').flatten()
    
#    plt.subplot(5,5,1+i)
#    plt.imshow(img, cmap='gray')
#    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
#plt.show()

#calculate the mean face
mean_face = np.zeros((1,height*width))
print("MeanFace")
print(mean_face)

for i in training_tensor:
    mean_face = np.add(mean_face,i)

mean_face = np.divide(mean_face,float(len(train_image_names))).flatten()

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
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

print("normalised:")
data = normalised_training_tensor
a = (data.T)
print("A = ")
print(a)
print("type:")
print(type(a))
b = data

plt.imshow(a)
plt.show()

ATA = np.mat(b) * np.mat(a)
AAT = np.mat(a) * np.mat(b)
print("ATA")
print(ATA)
#print("AAT")
#print(AAT)

eigenValues, eigenVectors = la.eig(ATA)

print(len(eigenValues))
print(len(eigenVectors))
print("eigenVector")
print(eigenVectors)
print("eigenValues")
for i in range(len(eigenValues)):
    print(eigenValues[i])

Av = []
AvNorm = []

for i in range(len(eigenVectors)):
    temp =  a* (eigenVectors.T[i].T)
    avTemp = temp.T
    normTemp = la.norm(avTemp)
    print("calculate norm u["+str(i)+"]:")
    Av.append (temp)
    AvNorm.append(avTemp/normTemp)


#p0 
p0 = normalised_training_tensor[0].reshape(-1,1)
p1 = normalised_training_tensor[1].reshape(-1,1)
p2 = normalised_training_tensor[2].reshape(-1,1)
p3 = normalised_training_tensor[3].reshape(-1,1)


uinorm0 = AvNorm[0]
uinorm1 = AvNorm[1]
uinorm2 = AvNorm[3]

#p
print("1")
w0 = uinorm0 * p0
w1 = uinorm1 * p0
w2 = uinorm2 * p0
print(w0)
print(w1)
print(w2)

print("2")
w0 = uinorm0 * p1
w1 = uinorm1 * p1
w2 = uinorm2 * p1
print(w0)
print(w1)
print(w2)

print("3")
w0 = uinorm0 * p2
w1 = uinorm1 * p2
w2 = uinorm2 * p2
print(w0)
print(w1)
print(w2)

print("4")
w0 = uinorm0 * p3
w1 = uinorm1 * p3
w2 = uinorm2 * p3
print(w0)
print(w1)
print(w2)



'''


# calculate covariance matrix
cov_matrix=np.cov(normalised_training_tensor)
print("COV_MATRIX")
print(cov_matrix)
#cov_matrix = np.divide(cov_matrix,25.0)
print('Covariance Matrix Shape:', cov_matrix.shape)
#print('Covariance matrix of X: \n%s' %cov_matrix)

#eigenvalues and eigenvectors
eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
print('eigenvalues.shape: {} eigenvectors.shape: {}'.format(eigenvalues.shape, eigenvectors.shape))
print("eigenVectorX")
for i in range(len(eigenvectors)):
    print(eigenvectors[i])
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


print("eigenvectors_sort")
for i in range(len(eigenVectors)):
    print(eigenVectors[i])

print("eigvectors_sort")
for i in range(len(eigvectors_sort)):
    print(eigvectors_sort[i])

eigenVectors_sort = eigenVectors[sorted_ind]

print("eigenVectors_sort ------ real")
for i in range(len(eigenVectors_sort)):
    print(eigenVectors_sort[i])

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
print("reduced:")
print(reduced_data)
print(reduced_data.shape)
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
'''