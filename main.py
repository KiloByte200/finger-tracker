import cv2
import numpy as np
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


model_path = "hand_landmarker.task"
num_hands = 1
running_mode = VisionRunningMode.VIDEO
connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

base_options = BaseOptions(
    model_asset_path = model_path
)

options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=num_hands,
    running_mode=running_mode
)


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
    upper = (30, 190, 220)

    camera = cv2.VideoCapture(0)

    cv2.namedWindow("Webcam")
    cv2.namedWindow("Webcam2")

    cv2.setMouseCallback("Webcam2", mouse_event)

    with HandLandmarker.create_from_options(options) as landmarker:
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
            #cv2.imshow("Webcam2", frame)

            contour_view(mask=mask, 
                        mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=RGB
            )

            timestamp_ms = int(time.monotonic() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                height, width, channels = frame.shape
                hand = result.hand_landmarks[0]

                for connection in connections:
                    start_landmark = hand[connection.start]
                    end_landmark = hand[connection.end]
            
                    start_x = int(width * start_landmark.x)
                    start_y = int(height * start_landmark.y)
                                    
                    end_x = int(width * end_landmark.x)
                    end_y = int(height * end_landmark.y)
                
                    cv2.line(
                                img=frame,
                                pt1=(start_x, start_y),
                                pt2=(end_x, end_y),
                                color=(255, 0, 0),
                                thickness=2
                            )

                for landmark in result.hand_landmarks[0]:
                    pixel_x = int(landmark.x * width)
                    pixel_y = int(landmark.y * height)

                    cv2.circle(
                        img=frame, 
                        center=(pixel_x, pixel_y),
                        radius=10,
                        color=(0, 255, 0),
                        thickness=-1
                        )


            cv2.imshow("Webcam2", frame)




    cv2.destroyAllWindows()
    camera.release()

if __name__ == "__main__":
    main()