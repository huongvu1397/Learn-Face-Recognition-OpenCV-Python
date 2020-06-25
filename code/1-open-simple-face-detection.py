import cv2

def draw_boundary(img,classifier,scaleFactor ,minNeighbors,color,text):
    # chuyển ảnh về xám
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # khởi tạo đối tượng face_detector
    face_detector = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors)
    coords = []
    for(x,y,w,h) in face_detector:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img,faceCasade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCasade, 1.1, 10, color['blue'], "Face")
    return img

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    img = detect(img,face_cascade)
    cv2.imshow("face_",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()