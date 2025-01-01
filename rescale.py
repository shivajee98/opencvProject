import cv2 as cv

img = cv.imread('../../m14.jpg')
cv.imshow('Baby-shot Image', img)

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
