import cv2

#camera laptop
rec = cv2.VideoCapture(0)
faceDetector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
tempCount = 0
while(True):
    ret, frame = rec.read()
    #flip frame
    frame = cv2.flip(frame,1)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = faceDetector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #draw rect
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        tempCount = tempCount + 1
        #write img
        cv2.imwrite("./step1/data/people_"+str(tempCount)+".jpg",gray[y:y+h,x:x+w])
 
    cv2.imshow('frame',frame)
    if cv2.waitKey(100) & 0xFF == ord('q') :
        break
    elif tempCount> 10:
        break

rec.release()
cv2.destroyAllWindows()  

