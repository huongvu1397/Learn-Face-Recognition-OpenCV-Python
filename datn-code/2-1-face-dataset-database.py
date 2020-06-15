import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Insert hoặc Update CSDL
def insertOrUpdate(id,name):
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

id = input('Nhập STT: ')
name = input('Nhập tên người: ')
print('Bắt đầu ghi lại khuôn mặt của người, nhấn q để thoát!')

insertOrUpdate(id,name)

sampleNum = 0
while(True):
    ret,img = cam.read()
    #lật ảnh
    img = cv2.flip(img,1)

    #đưa ảnh về xám
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #phát hiện khuôn mặt
    faces = detector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #Vẽ hình chữ nhật quanh mặt nhận được
        cv2.rectangle(img , (x,y) ,( x+w , y+h ),(255,0,0),2)
        sampleNum = sampleNum + 1
        #Ghi dữ liệu khuôn mặt vào thư mục
        cv2.imwrite("dataset/User."+id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

    cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif sampleNum>100:
        break

cam.release()
cv2.destroyAllWindows()


