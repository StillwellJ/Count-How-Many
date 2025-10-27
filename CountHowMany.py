import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromONNX("best.onnx")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
img = cv.imread("test.jpg")

height, width = img.shape[:2]

blob = cv.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)
detections = net.forward()
detections = detections.transpose()

count = 0
for detection in detections:
    confidence = float(detection[5])
    if confidence > 0.5:
        count += 1

        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)

        angle = float(detection[4])

        rect = ((center_x, center_y), (w, h), np.degrees(angle))
        box = cv.boxPoints(rect)
        box = np.intp(box)
        cv.drawContours(img, [box], 0, (0, 255, 0), 2)

        label = f"{confidence:.2f}"
        cv.putText(img, label, (center_x - 20, center_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print("Car count: ", count)
cv.imshow("Detections", img)
cv.waitKey(0)
cv.destroyAllWindows()