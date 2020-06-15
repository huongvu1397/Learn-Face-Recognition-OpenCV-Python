import cv2
import numpy as np
from PIL import Image
import os

#Thư mục chứa tập dữ liệu training.
path = "dataset"

#Bộ nhận dạng
recognizer = cv2.face.EigenFaceRecognizer_create(10,3000)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Hàm lấy ảnh và nhãn (images & label)
def getImagesAndLabels(path):
    height_d,width_d = 100, 100

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples= []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces : 
            faceSamples.append(cv2.resize(img_numpy[y:y+h,x:x+w],(height_d,width_d)))
            ids.append(id)

    return faceSamples,ids

print ("\n Đang huấn luyện khuôn mặt ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu model tại thư mục trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

print("\n Đã huấn luyện thành công {0} khuôn mặt . Kết thúc chương trình".format(len(np.unique(ids))))
