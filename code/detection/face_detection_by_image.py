import cv2

file_name = 'test_trump'
path = "./data/face_rec_test/"+file_name+".jpg"

size = 100
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#image
img = cv2.imread(path)
mini = cv2.resize(img,(size,size))
gray = cv2.cvtColor(mini,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.1,3)

print("count face : ",len(faces))
tempCount = 0
for(x,y,w,h) in faces :
    cv2.rectangle(mini,(x,y),(x+w,y+h),(255,0,0),3) 
    tempCount = tempCount + 1
    resize_image = cv2.resize(img[y:y+h,x:x+w],(size,size))
    cv2.imwrite("./data/face_rec_test/test_"+file_name+"_"+str(tempCount)+".jpg",resize_image)

#for image
cv2.imshow('img',resize_image)
cv2.waitKey()