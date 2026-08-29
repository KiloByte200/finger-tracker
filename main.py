import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        print("Unsuccessful camera capture")
    print(success)

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("Webcam", RGB)
    cv2.waitKey(1)