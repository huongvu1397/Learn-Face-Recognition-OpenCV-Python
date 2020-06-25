import cv2
import numpy as np
from PIL import Image
import os

#Thư mục chứa tập dữ liệu training.
path = "datn-dataset"

#Khởi tạo bộ nhận dạng
# Với tham số đầu tiên là num_components : Số lượng thành phần được giữ cho PCA. 
# Gợi ý: không có quy tắc là bao nhiêu thành phần nên được giữ lại để có khả năng tái sử dụng tốt.
# Tham số thứ hai là threshold : Ngưỡng áp dụng trong dự đoán để xác định khuôn mặt.
recognizer = cv2.face.EigenFaceRecognizer_create(num_components= 0,threshold= 3000)

#Khởi tạo bộ phát hiện khuôn mặt
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Phân loại dữ liệu (images & label)
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

# Lưu model huấn luyện tại thư mục trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

print("\n Đã huấn luyện thành công {0} khuôn mặt . Kết thúc chương trình".format(len(np.unique(ids))))
