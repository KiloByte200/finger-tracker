import cv2
import numpy as np

latest_hsv = None

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(latest_hsv[y, x])

def main():
    global latest_hsv

    camera = cv2.VideoCapture(1)

    while True:
        success, frame = camera.read()
        if not success:
            print("Unsuccessful camera capture")
        #print(success)

        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        latest_hsv = HSV
        

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lower = (0, 60, 100)
        upper = (30, 140, 200)

        mask = cv2.inRange(HSV, lower, upper)

        cv2.imshow("Webcam", mask)
        cv2.imshow("Webcam2", frame)
        cv2.waitKey(1)

        cv2.setMouseCallback("Webcam2", mouse_event)

        contours, hierarchy = cv2.findContours(
            image=mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contour = max(contours, key=cv2.contourArea)
        contour_view = np.zeros_like(mask)

        cv2.drawContours(
            image=contour_view,
            contours=[largest_contour],
            contourIdx=-1,
            color=255,
            thickness=2)

        cv2.imshow("Contour_View", contour_view)

if __name__ == "__main__":
    main()