#thêm thư viện opencv
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Chụp từng khung hình
    ret, frame = cap.read()

    # Hiển thị khung hình chụp được
    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

#Kết thúc và giải phóng 'cap'
cap.release()
cv2.destroyAllWindows


    