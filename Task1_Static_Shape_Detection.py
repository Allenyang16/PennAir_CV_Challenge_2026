import cv2 as cv
import sys
import numpy as num

img = cv.imread(cv.samples.findFile("PennAir 2024 App Static.png"))
if img is None:
    sys.exit("Could not read the image.")

# print(img.shape)
# print(img.size)

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV) # Hue, Saturation, Value
# print(hsv.shape)

lower = (0, 50, 150)
upper = (179, 255, 255)

img2 = cv.inRange(hsv,lower,upper)

# cv.imshow("Original",img)
# cv.imshow("Mask", img2)
# cv.waitKey(0)
# cv.destroyAllWindows

contours, hierarchy = cv.findContours(
    img2,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

MIN_AREA = 100

for contour in contours:
    area = cv.contourArea(contour)
    if area < MIN_AREA :
        continue

    cv.drawContours(
        img,
        [contour],
        -1,
        [0,0,0],
        3
    )

    M = cv.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv.circle(img,(cx,cy),3,(0,0,0),-1)
    print("Center:", cx, cy)

cv.imshow("Result",img)
cv.waitKey(0)
cv.destroyAllWindows
