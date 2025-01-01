import cv2

img = cv2.imread('../../m14.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('img', img)

cv2.waitKey(1000)

cv2.destroyAllWindows()