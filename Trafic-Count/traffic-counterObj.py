from ultralytics import YOLO
import cv2
import cvzone
import math

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

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    width = 800
    height = 500

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
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)

    # Text on the screen
    cv2.putText(img, f"Traffic Counter: {currentClass}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # To resize the image
    resize_img = cv2.resize(img, (width, height))

    # FPS
    FPS = 1 / 30
    FPS_MS = int(FPS * 1000)

    cv2.imshow("Tracking", resize_img)
    cv2.waitKey(FPS_MS)