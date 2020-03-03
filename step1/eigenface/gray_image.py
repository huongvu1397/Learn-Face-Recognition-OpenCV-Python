import cv2 


#1.jpg
#./step1/data/1.jpg
#2.jpg
#./step1/data/2.jpg
#3.jpg
#./step1/data/3.jpg
#4.jpg
#./step1/data/4.jpg
#5.jpg
#./step1/data/5.jpg

image1 = cv2.imread('./step1/data/1.jpg')
image2 = cv2.imread('./step1/data/2.jpg')
image3 = cv2.imread('./step1/data/3.jpg')
image4 = cv2.imread('./step1/data/4.jpg')
image5 = cv2.imread('./step1/data/5.jpg')

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)
  

cv2.imwrite('./step1/data/gray_1.jpg',gray1)
cv2.imwrite('./step1/data/gray_2.jpg',gray2)
cv2.imwrite('./step1/data/gray_3.jpg',gray3)
cv2.imwrite('./step1/data/gray_4.jpg',gray4)
cv2.imwrite('./step1/data/gray_5.jpg',gray5)

