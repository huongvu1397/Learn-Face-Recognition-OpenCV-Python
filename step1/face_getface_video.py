import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#video
cap = cv2.VideoCapture('./step1/video/thuy.mp4')

tempCount = 0
while cap.isOpened():
    _,frame = cap.read()
    # flip vertical
    frame = cv2.flip(frame,0)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    sample = frame#cv2.cvtColor(frame,cv2.COLOR_)

    faces = face_cascade.detectMultiScale(gray,1.3,4)
    
    for(x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        tempCount = tempCount + 1
        resize_image = cv2.resize(sample[y:y+h,x:x+w],(100,100))
        cv2.imwrite("./step1/data/thuy/thuy_"+str(tempCount)+".jpg",resize_image)

    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #elif tempCount>150:
     #   break
cap.release()
cv2.destroyAllWindows() 
