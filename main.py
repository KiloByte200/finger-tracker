import cv2

latest_hsv = None

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(latest_hsv[y, x])

def main():
    global latest_hsv

    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            print("Unsuccessful camera capture")
        #print(success)

        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        latest_hsv = HSV
        

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lower = (0, 0, 180)
        upper = (30, 100, 255)

        mask = cv2.inRange(HSV, lower, upper)

        cv2.imshow("Webcam", mask)
        cv2.imshow("Webcam2", frame)
        cv2.waitKey(1)

        cv2.setMouseCallback("Webcam2", mouse_event)

if __name__ == "__main__":
    main()