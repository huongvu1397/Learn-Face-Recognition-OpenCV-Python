import cv2
import numpy as np
import os
import sqlite3


# Khởi tạo bộ nhận diện khuôn mặt
#load model huấn luyện tại chương trình 2-2
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

cascadePath = "haarcascade_frontalface_default.xml"
# Khởi tạo bộ phát hiện khuôn mặt
faceCascade = cv2.CascadeClassifier(cascadePath)

#Lấy thông tin trong database
def getPeople(id):
    conn=sqlite3.connect("FaceUserDatabase.db")
    cursor=conn.execute("SELECT * FROM people WHERE ID="+str(id))
    people=None
    for row in cursor:
        names=row
    conn.close()
    return names

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    img = cv2.flip(img,1)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        roi = gray[y:y+h,x:x+w]

        name = None

        try:
            roi = cv2.resize(roi,(100,100))
            predictedLabel,confidence = recognizer.predict(roi)
            if(predictedLabel  == -1):
                print("Label : %s , Confidence : %.2f    ",predictedLabel,confidence)
                cv2.putText(img,"unknown",(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
            else:
                name = getPeople(predictedLabel)
                print("Label : %s , Confidence : %.2f    ",predictedLabel,confidence)
                cv2.putText(img,str(name[0]),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

        except: 
            continue
    cv2.imshow("camera",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n Kết thúc chương trình")
cam.release()
cv2.destroyAllWindows()
