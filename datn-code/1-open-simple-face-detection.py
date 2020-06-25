import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    #chụp khung hình trên camera
    _, img = cap.read()
    #làm xám khung hình chụp được
    gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #khởi tạo bộ phát hiện khuôn mặt trên khung hình chụp được trên camera
    face_detector = face_cascade.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=4)
    #quét khuôn mặt
    for(x,y,w,h) in face_detector:
        #vẽ hình vuông và chữ lên khuôn mặt tìm được
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        cv2.putText(img, "Face", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)


    #hiển thị khung hình thu được trên camera
    cv2.imshow("Face Detection",img)

    #bấm Q để tắt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()