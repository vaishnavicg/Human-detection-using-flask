import numpy as np
import cv2
import cv2 as cv
from ultralytics import YOLO

def video_detection(path_x):
    video_path=path_x
    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # Initialize frame counter
    model = YOLO('best.pt')
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1  # Increment the frame counter
        
        if frame_count % 20 != 0:  # Skip all frames except every 10th frame
            continue

        frame = cv2.resize(frame,(640,640))
        height, width, _ = frame.shape
        
        results = model.predict(frame)
        for result in results:
            n_detections = result.boxes.shape[0]
            if(n_detections != 0):
                print(result.boxes.data.shape[0], result.boxes.data.dtype)
                k_model = (np.asarray(result.boxes.data[0])[:4]).flatten().astype(np.int32)
                print("k-model ", k_model)

                roi = frame[k_model[1]:k_model[3], k_model[0]:k_model[2],:]
                track_window = (k_model[0],k_model[1], k_model[2]-k_model[0], k_model[3]-k_model[1])

                hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
                roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
                cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

                term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                
                ret, track_window = cv.CamShift(dst, track_window, term_crit)
                
                # print("DETECTED")
                print('return = ',ret)
                print("Final_Track_Window= ", track_window)

                pts = cv.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv.polylines(frame, [pts], True, 255, 2)
                cv2.putText(img2, 'DETECTED', (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
            else:
                
                # print("KOI NAHI HAI")
                cv2.putText(frame, 'NON DETECTED', (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
                img2 = frame

        # print('I am running')
        #cv2.imshow('img2', frame)
        yield img2

        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     cv2.destroyAllWindows()
        #     break

# Load the YOLOv8 model
# model = YOLO('best.pt')

# # Open the video file
# video_path = "part4.mp4"
# cap = cv2.VideoCapture(video_path)
# tracking()