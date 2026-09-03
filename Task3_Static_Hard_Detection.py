import cv2 as cv
import sys
import numpy as num

image = cv.imread(cv.samples.findFile("Hard_Static.png"))
if image is None:
    sys.exit("Could not read the image.")

pentagon_parts = []

def detect_shape(img):
    centers = []
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV) # Hue, Saturation, Value
    # 条件 1：比较亮的东西
    bright_mask = cv.inRange(
        hsv,
        (0, 0, 100),
        (179, 255, 255)
    )

    # 条件 2：虽然不亮，但颜色足够鲜艳
    color_mask = cv.inRange(
        hsv,
        (0, 100, 50),
        (179, 255, 255)
    )
    img2 = cv.bitwise_or(bright_mask,color_mask)
    # lower = (0, 0, 100) #现在主要是亮度的限制，亮度范围高才能进入
    # upper = (179, 255, 255)

    # img2 = cv.inRange(hsv,lower,upper)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    erosion = cv.erode(img2,kernel,iterations = 1)
    opening = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(img2, cv.MORPH_CLOSE, kernel)
    cv.namedWindow("Mask", cv.WINDOW_NORMAL)
    cv.imshow("Mask", img2)
    # cv.namedWindow("Opening", cv.WINDOW_NORMAL)
    # cv.imshow("Opening", opening)
    # cv.namedWindow("Closing", cv.WINDOW_NORMAL)
    # cv.imshow("Closing", closing)
    
    contours, hierarchy = cv.findContours(
        img2,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    MIN_AREA = 1000

    for contour in contours:
        area = cv.contourArea(contour)
        if area < MIN_AREA :
            continue

        print("Area",area)
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
        if cx > 1200 and cy > 600:
            pentagon_parts.append(contour)
        cv.circle(img,(cx,cy),3,(0,0,0),-1)
        centers.append((cx, cy))
        print("Center:", cx, cy)

    return img

def merge_for_pentagon(img):
    if len(pentagon_parts) == 2:
        combined = num.vstack(pentagon_parts)

        hull = cv.convexHull(combined)

        # cv.drawContours(
        #     img,
        #     [hull],
        #     -1,
        #     (255, 255, 255),
        #     5
        # )

        M = cv.moments(hull)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        print("Merged pentagon center:", cx, cy)
        perimeter = cv.arcLength(hull, True)
        approx = cv.approxPolyDP(hull, 0.04 * perimeter, True)
        print(len(approx))
        cv.drawContours(
            img,
            [approx],
            -1,
            (255, 255, 255),
            5
        )

    return img

result = detect_shape(image)
print("pentagon",len(pentagon_parts))
result = merge_for_pentagon(result)
cv.namedWindow("Result", cv.WINDOW_NORMAL)
cv.imshow("Result", result)
cv.waitKey(0)
cv.destroyAllWindows()

