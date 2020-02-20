import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#image
img = cv2.imread('./group_people.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.1,5)
tempCount = 0
for(x,y,w,h) in faces :
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3) 
    tempCount = tempCount+1
    cv2.imwrite("./step1/data/img_"+str(tempCount)+".jpg",gray[y:y+h,x:x+w])

#for image
cv2.imshow('img',img)
cv2.waitKey()

