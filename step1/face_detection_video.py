import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#video
cap = cv2.VideoCapture('video-low.mp4')
tempCount = 0
while cap.isOpened():
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
  
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        tempCount = tempCount + 1
        cv2.imwrite("./step1/data/fromVid_"+str(tempCount)+".jpg",gray[y:y+h,x:x+w])

    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #elif tempCount>150:
     #   break
cap.release()
cv2.destroyAllWindows() 
