import cv2 as cv
import numpy as np


net = cv.dnn.readNetFromONNX("best.onnx")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def run_model(frame):
    """
    Runs best.onnx to make detections on a given frame

    Parameters:
    frame - frame to make detections on

    Output:
    detections - the detections that openCV made on the image
    height - height of the image
    width - width of the image
    """
    height, width = frame.shape[:2]

    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    detections = detections.transpose()

    return detections


def counter():
    """
    Counts how many cars are in front of the camera
    """
    cap = cv.VideoCapture(0)
    window_name = "Car Detection"
    cv.namedWindow(window_name)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera not found.")
            break

        detections = run_model(frame)
        count = 0
        for detection in detections:
            confidence = float(detection[5])
            if confidence > 0.5:
                count += 1
                center_x = int(detection[0])
                center_y = int(detection[1])
                w = int(detection[2])
                h = int(detection[3])
                angle = float(detection[4])

                rect = ((center_x, center_y), (w, h), np.degrees(angle))
                box = cv.boxPoints(rect)
                box = np.intp(box)
                cv.drawContours(frame, [box], 0, (0,255,0), 2)
                label = f"{confidence:.2f}"
                cv.putText(frame, label, (center_x - 20, center_y -10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        cv.setWindowTitle(window_name, f"Car Detection | Count: {count}")
        cv.imshow(window_name, frame)

        if cv.waitKey(1) == 27: #ESC key to quit
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    counter()
