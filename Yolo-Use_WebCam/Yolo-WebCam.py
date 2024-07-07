from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(3, 1080)

model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
              "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant",
              "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
              "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

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
            cvzone.cornerRect(img, (x1,y1,w,h))

            """
            Label confidences  
            """
            conf = math.ceil((box.conf[0]*100))/100

            """
            Class Name
            """
            cls = int(box.cls[0])
            print()

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35, y1)))

    cv2.imshow("Tracking", img)
    cv2.waitKey(1)