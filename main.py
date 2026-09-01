import cv2
import numpy as np
import mediapipe as mp

latest_hsv = None

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and latest_hsv is not None:
        print(latest_hsv[y, x])

def contour_view(mask: cv2.typing.MatLike, 
                 mode: int,
                 method: int) -> None:
    

    contours, hierarchy = cv2.findContours(
                image=mask,
                mode=mode,
                method=method
            )
    
    if not contours:
        contour_view = np.zeros_like(mask)
        cv2.imshow("Contour_View", contour_view)
        return
        

    largest_contour = max(contours, key=cv2.contourArea)
    contour_view = np.zeros_like(mask)
    
    cv2.drawContours(
        image=contour_view,
        contours=[largest_contour],
        contourIdx=-1,
        color=255,
        thickness=2)
    
    cv2.imshow("Contour_View", contour_view)

     

def main():
    global latest_hsv

    lower = (0, 30, 100)
    upper = (30, 140, 220)

    camera = cv2.VideoCapture(0)

    cv2.namedWindow("Webcam")
    cv2.namedWindow("Webcam2")

    cv2.setMouseCallback("Webcam2", mouse_event)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        success, frame = camera.read()
        if not success:
            print("Unsuccessful camera capture")
            break
        #print(success)

        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        latest_hsv = HSV

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = cv2.inRange(HSV, lower, upper)

        cv2.imshow("Webcam", mask)
        cv2.imshow("Webcam2", frame)

        contour_view(mask=mask, 
                     mode=cv2.RETR_EXTERNAL,
                     method=cv2.CHAIN_APPROX_SIMPLE)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=RGB
        )




    cv2.destroyAllWindows()
    camera.release()

if __name__ == "__main__":
    main()