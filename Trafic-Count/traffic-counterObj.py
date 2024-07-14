import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../videos/2.mp4")

model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
              "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant",
              "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
              "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsOut = [100,750,850,750]
limitsIn = [1050,750,1750,750]
totCountIn = []
totCountOut = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections =np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:

            """
            ini buat box
            """
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (254, 4, 4), 3)

            """
            ini adalah buat corner pada box
            """
            w, h = x2-x1, y2-y1
            # bbox = int(x1), int(y1), int(w), int(h)

            """
            Label confidences  
            """
            conf = math.ceil((box.conf[0]*100))/100

            """
            Class Name
            """
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img,(limitsOut[0],limitsOut[1]),(limitsOut[2],limitsOut[3]),(0,0,255), 5)
    cv2.line(img, (limitsIn[0], limitsIn[1]), (limitsIn[2], limitsIn[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1+w//2,y1+h//2
        cv2.circle(img, (cx,cy),5,(255.0,255),cv2.FILLED)

        if limitsOut[0]<cx<limitsOut[2] and limitsOut[1]-20<cy<limitsOut[3]+20:
            if totCountOut.count(id) ==0:
                totCountOut.append(id)

        if limitsIn[0]<cx<limitsIn[2] and limitsIn[1]-20<cy<limitsIn[3]+20:
            if totCountIn.count(id) ==0:
                totCountIn.append(id)

        cvzone.putTextRect(img, f' Keluar: {len(totCountOut)}', (50, 50))
        cvzone.putTextRect(img, f' Masuk: {len(totCountIn)}', (1600,50))

    cv2.imshow("Tracking", img)
    cv2.waitKey(1)