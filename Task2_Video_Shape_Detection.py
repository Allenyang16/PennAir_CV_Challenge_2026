import cv2 as cv
import sys
import numpy as num

def detect_shape(img):
    centers = []
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV) # Hue, Saturation, Value

    lower = (0, 50, 150)
    upper = (179, 255, 255)

    img2 = cv.inRange(hsv,lower,upper)

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
        centers.append((cx, cy))
        cv.putText(
            img,
            "center",
             (int(cx)-15, int(cy)+30),
             cv.FONT_HERSHEY_SIMPLEX,
             0.7,
             (75, 255, 100),
             1
        )
        cv.putText(
            img,
            f"({cx:.1f}, {cy:.1f})",
            (int(cx)-15 , int(cy)+60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (75, 255, 100),
            1
        )
        # print("Center:", cx, cy)
    return img, centers

video = cv.VideoCapture("PennAir 2024 App Dynamic.mp4")
if not video.isOpened():
    print("Cannot open camera")
    exit()
while video.isOpened:
    ret,frame = video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    result, centers = detect_shape(frame)
    cv.imshow("Result", result)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv.destroyAllWindows()

# For this challenge, what should I do if the two countours overlap up with each other? 
# Because there are sometimes where the original center is being blocked 
# S