import cv2
import os

#thư mục ảnh
IMG_FOLDER = './step1/imgtest/'
SAVE_FOLDER = './code/convert_gray_image/data/1/'


img_list = os.listdir(IMG_FOLDER)

for i in range(len(img_list)):
    img = cv2.imread(IMG_FOLDER + img_list[i])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(SAVE_FOLDER+str(i)+".jpg",gray)
    

