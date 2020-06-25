import cv2
import sqlite3
import os
import datetime
import sys
import numpy as np
from PIL import Image

#https://iosoft.blog/2019/07/31/rpi-camera-display-pyqt-opencv/
#https://www.youtube.com/watch?v=iA45JnQh3Ow

from PyQt5 import QtCore,QtGui,QtWidgets, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication,QDialog,QMainWindow,QFileDialog

class MainFaceTraining(QMainWindow):
    def __init__(self):
        super(MainFaceTraining,self).__init__()
        uic.loadUi('datn-gui/huan_luyen_gui.ui',self)
        self.btnStart.clicked.connect(self.onClickBtnStart)
        self.btnExit.clicked.connect(self.onClickBtnExit)
        self.btnTraining.clicked.connect(self.onClickBtnTraining)
        self.textBrowser.setText('Điền thông tin người cần nhận dạng, ra trước camera và chọn bắt đầu để thực hiện việc thu thập hình ảnh khuôn mặt.')

    @pyqtSlot()
    def onClickBtnStart(self):
        text_name = self.edtName.toPlainText()
        text_id = self.edtId.toPlainText()
        convertIdToInt = int(text_id)
        self.textBrowser.setText('Thêm '+text_name+' vào cơ sở dữ liệu nhận dạng. Tiến hành thêm khuôn mặt...')
        self.insertOrUpdate(text_id,text_name)

        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        
        sampleNum = 0
        while True:

            ret, img =cam.read()
            #lật ảnh
            img = cv2.flip(img,1)

            #đưa ảnh về xám
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #phát hiện khuôn mặt
            faces = detector.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5)

            for(x,y,w,h) in faces:
                #Vẽ hình chữ nhật quanh mặt nhận được
                cv2.rectangle(img , (x,y) ,( x+w , y+h ),(255,0,0),2)
                sampleNum = sampleNum + 1
                #Ghi dữ liệu khuôn mặt vào thư mục
                if(sampleNum < 100):
                    pathImage = r'C:\Users\Huong Vu\Desktop\GitDATN\Learn-Face-Recognition-OpenCV-Python\datn-dataset/User.%s.%s.jpg'%(str(id),str(sampleNum))
                    #cv2.imwrite(pathImage,gray[y:y+h,x:x+w])
                    path = "datn-dataset/User."+str(text_id)+'.'+str(sampleNum)+".jpg"
                    print("Write image ",path)
                    cv2.imwrite("datn-dataset/User."+str(text_id)+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

            self.displayImage(img)

            if cv2.waitKey() & 0xFF == ord('q'):
                break
            elif sampleNum>100:
                self.textBrowser.setText('Kết thúc quá trình thu thập hình ảnh. Có thể chọn "Huấn luyện" để huấn luyện bộ dữ liệu khuôn mặt.')
                break

        #cam.release()
        #cv2.destroyAllWindows()

    def displayImage(self,img):
        if img is not None :
            qformat = QImage.Format_Indexed8
            if len(img.shape)== 3:
                if(img.shape[2]) == 4:
                    qformat = QImage.Format_RGBA888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(img,img.shape[1],img.shape[0],qformat)
            img = img.rgbSwapped()
            self.videoFrame.setPixmap(QPixmap.fromImage(img))
            self.videoFrame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def onClickBtnExit(self):
        cam.release()
        cv2.destroyAllWindows()
        widget.close()

    @pyqtSlot()
    def onClickBtnTraining(self):
        self.trainingImage()

    def trainingImage(self):
        #Thư mục chứa tập dữ liệu training.
        path = "datn-dataset"
        #threshold = self.edtThreshold.toPlainText()
        #Khởi tạo bộ nhận dạng # 0 component là lấy hết thành phần của eigenfaces
        recognizer = cv2.face.EigenFaceRecognizer_create(num_components= 0,threshold= 3000)
        #Khởi tạo bộ phát hiện khuôn mặt
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        print ("\n Đang huấn luyện khuôn mặt ...")
        faces,ids = self.getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        # Lưu model huấn luyện tại thư mục trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')
        print("\n Đã huấn luyện thành công {0} khuôn mặt . Kết thúc chương trình".format(len(np.unique(ids))))

    #Phân loại dữ liệu (images & label)
    def getImagesAndLabels(self,path):
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



    # Insert hoặc Update CSDL
    def insertOrUpdate(self,id,name):
        conn = sqlite3.connect("FaceUserDatabase.db")
        cursor = conn.execute('SELECT * FROM people WHERE ID='+str(id))
        isRecordExist = 0
        for row in cursor:
            isRecordExist = 1
            break
        if isRecordExist == 1:
            cmd = "UPDATE people SET Name='"+str(name)+"' WHERE ID ="+str(id)
        else:
            cmd = "INSERT INTO people(ID,Name) Values("+str(id)+",' "+str(name)+"')"
    
        conn.execute(cmd)
        conn.commit()
        conn.close()

#Init
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



app = QApplication(sys.argv)
widget = MainFaceTraining()
widget.show()

try:
    sys.exit(app.exec_())
except:
    print('exiting')
