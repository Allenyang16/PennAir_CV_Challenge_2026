import cv2 as cv
import sys
import numpy as num

def detect_shape(img):
    centers = []
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV) # Hue, Saturation, Value

    # lower = (0, 0, 100)
    # upper = (179, 255, 255)
    # img2 = cv.inRange(hsv,lower,upper)
    bright_mask = cv.inRange(
            hsv,
            (0, 0, 100),
            (179, 255, 255)
        )
    color_mask = cv.inRange(
            hsv,
            (0, 100, 0),
            (179, 255, 255)
        )
    img2 = cv.bitwise_or(bright_mask,color_mask)
    

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

video = cv.VideoCapture("PennAir 2024 App Dynamic Hard.mp4")
paused = False
frame = None
if not video.isOpened():
    print("Cannot open camera")
    exit()
while video.isOpened:
    if not paused:
        ret,current_frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame, centers = detect_shape(current_frame)

    if frame is not None: 
        cv.imshow("Result", frame)

    # Wait for keypress (30ms interval)
    key = cv.waitKey(30) & 0xFF
    
    # Press 'Spacebar' to toggle freeze/unfreeze
    if key == ord(' '):
        paused = not paused
        
    # Press 'q' to quit
    elif key == ord('q'):
        break

video.release()
cv.destroyAllWindows()

