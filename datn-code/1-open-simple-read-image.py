import cv2

# Load an image
img = cv2.imread('opencv.png')
cv2.imshow("read image",img)
cv2.waitKey(5000)
print("Huong vv")
cv2.destroyAllWindows()