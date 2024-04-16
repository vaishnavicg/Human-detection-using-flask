# from ultralytics import YOLO
# import cv2
# import math

# def video_detection(path_x):
#     video_capture = path_x
#     #Create a Webcam Object
#     cap=cv2.VideoCapture(video_capture)
#     frame_width=int(cap.get(3))
#     frame_height=int(cap.get(4))
#     #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

#     model=YOLO("best.pt")
    
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         results=model(img,stream=True)
#         for r in results:
#             boxes=r.boxes
#             for box in boxes:
#                 x1,y1,x2,y2=box.xyxy[0]
#                 x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
#                 print(x1,y1,x2,y2)
#                 cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                
#         yield img
#         #out.write(img)
#         #cv2.imshow("image", img)
#         #if cv2.waitKey(1) & 0xFF==ord('1'):
#             #break
#     #out.release()
# cv2.destroyAllWindows()







from ultralytics import YOLO
import cv2

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    model = YOLO("best.pt")
    
    frame_counter = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        frame_counter += 1
        
        if frame_counter % 10 == 0:  # Process every 10th frame
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(x1, y1, x2, y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            yield img
        
    cap.release()
    cv2.destroyAllWindows()
