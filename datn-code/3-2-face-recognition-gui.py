import cv2
import numpy as np
import os 
import datetime
import sys
import sqlite3

#https://iosoft.blog/2019/07/31/rpi-camera-display-pyqt-opencv/
#https://www.youtube.com/watch?v=iA45JnQh3Ow

from PyQt5 import QtCore,QtGui,QtWidgets, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication,QDialog,QMainWindow,QFileDialog

class MainFaceRecognition(QMainWindow):
    def __init__(self):
        super(MainFaceRecognition,self).__init__()
        uic.loadUi('datn-gui/nhan_dien_gui.ui',self)
        #event nút 'Bắt đầu'
        self.btnStart.clicked.connect(self.onClickBtnStart)
        #event nút 'Thoát'
        self.btnExit.clicked.connect(self.onClickBtnExit)
        #event nút 'Chụp hình'
        self.btnCapture.clicked.connect(self.onClickedCapture)
        #event nút 'Ghi hình'
        self.btnRecord.clicked.connect(self.onClickedRecord)
        self.textBrowser.setText('Ấn bắt đầu để chạy chương trình nhận diện khuôn mặt.')
        self.logic_capture = 0
        self.logic_record = 0
        self.fps = 24.0
        self.width = 640
        self.height = 480

    #Lấy thông tin trong database
    def getPeople(self,id):
        conn=sqlite3.connect("FaceUserDatabase.db")
        cursor=conn.execute("SELECT * FROM people WHERE ID="+str(id))
        people=None
        for row in cursor:
            people=row
        conn.close()
        return people

    @pyqtSlot()
    def onClickBtnStart(self):
        self.textBrowser.setText('Bắt đầu chạy.')
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height

        while True:

            ret, img =cam.read()
            img = cv2.flip(img,1)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5)

            names = None

            for(x,y,w,h) in faces:
                img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                roi = gray[y:y+h,x:x+w]
                try:
                    roi = cv2.resize(roi,(100,100))
                    predictedLabel,confidence = recognizer.predict(roi)

                    if(predictedLabel  == -1):
                        #print("Label : %s , Confidence : %.2f    ",predictedLabel,confidence)
                        cv2.putText(img,"unknown",(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
                        self.textBrowser.setText('Phát hiện khuôn mặt của người lạ')

                    else:
                        names = self.getPeople(predictedLabel) #  [0] - tên [1]- id
                        print("name : ",names)
                        cv2.putText(img,str(names[0]),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
                        #print("Label : %s , Confidence : %.2f    ",predictedLabel,confidence)
                        txt_predicted = str(names[0])
                        txt_notification = "Phát hiện khuôn mặt của " + txt_predicted
                        self.textBrowser.setText(txt_notification)
                        
                    self.showInfoPerson(predictedLabel)


                except: 
                    continue
            #cv2.imshow("camera",img)
            self.displayImage(img,1)
            
            cv2.waitKey()
            
            #Chụp hình tại frame chỉ định
            if(self.logic_capture == 2):
                date = datetime.datetime.now()
                NAME = r'datn-image-capture/Image_%s%s%sT%s%s%s.png'%(date.year,date.month,date.day,date.hour,date.minute,date.second)
                print('Lưu hình ảnh tại :',NAME)
                cv2.imwrite(NAME,img)
                self.logic_capture = 1
            
            if(self.logic_record == 2):
                self.record.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("\n Kết thúc chương trình")
        cam.release()
        cv2.destroyAllWindows()

    def displayImage(self,img,window = 1):
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

    def onClickedCapture(self):
        self.logic_capture = 2

    def showInfoPerson(self,predictedLabel):
        if predictedLabel == -1:
            path_image = r'datn-info-person\unknown_image.jpg'
            self.imageFrame.setPixmap(QPixmap(path_image))
            self.imageFrame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.txtName.setText("Unknown")

        else:    
            path_image = r'datn-dataset\User.%s.1.jpg'%(predictedLabel)
            #qformat = QImage.Format_Indexed8
            #if len(img.shape)== 3:
            #    if(img.shape[2]) == 4:
            #        qformat = QImage.Format_RGBA888
            #    else:
            #        qformat = QImage.Format_RGB888
            #img = QImage(img,img.shape[1],img.shape[0],qformat)
            #img = img.rgbSwapped()
            self.imageFrame.setPixmap(QPixmap(path_image))
            self.imageFrame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.txtName.setText("Vũ Văn Hưởng")

    @pyqtSlot()
    def onClickedRecord(self):
        if self.logic_record == 0:
            #start record
            self.btnRecord.setText("Dừng ghi hình")
            cam.set(3, self.width) # set video widht
            cam.set(4, self.height) # set video height
            self.video_type = cv2.VideoWriter_fourcc(*'XVID')
            date = datetime.datetime.now()
            self.filename = r'datn-video-record/Video_%s%s%sT%s%s%s.avi'%(date.year,date.month,date.day,date.hour,date.minute,date.second)
            self.record = cv2.VideoWriter(self.filename,self.video_type,self.fps,(self.width,self.height))
            self.logic_record = 2

        elif self.logic_record == 2:
            #stop record
            self.btnRecord.setText("Ghi hình")
            self.logic_record = 0
            self.record.release()



    @pyqtSlot()
    def onClickBtnExit(self):
        self.record.release()
        cam.release()
        cv2.destroyAllWindows()
        widget.close()

#load model huấn luyện tại chương trình 2-2
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')   
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)


app = QApplication(sys.argv)
widget = MainFaceRecognition()
widget.show()

try:
    sys.exit(app.exec_())
except:
    print('exiting')
