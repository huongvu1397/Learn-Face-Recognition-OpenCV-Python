import cv2
import os

#thư mục ảnh
IMG_FOLDER = './data/imgtest_5/'
SAVE_FOLDER = './datn-helper/temp-helper/convert_gray_image/'


img_list = os.listdir(IMG_FOLDER)
print("list : ",len(img_list))
print("list : ",img_list[0])


for i in range(len(img_list)):
    img = cv2.imread(IMG_FOLDER + img_list[i])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(SAVE_FOLDER+str(i)+".jpg",gray)
    

