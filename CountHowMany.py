import cv2 as cv


net = cv.dnn.readNetFromONNX("best.onnx")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def run_model(frame):
    """
    Runs best.onnx to make detections on a given frame

    Parameters:
    frame - frame to make detections on

    Output:nump
    detections - the detections that openCV made on the image
    height - height of the image
    widtheight- width of the image
    """
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    return detections[0]

def process(detections, confidence_threshold=0.5, nms_threshold=0.45):
    """
    Processes detections and creates boxes based on a given
    confidence threshold.

    Parameters:
    detections - all of the detections made on a current frame
    confidence_threshold - the confidence threshold, defaults to 0.5
    nms_threshold - non max suppression threshold, threshold used to stop box stacking, defaults to 0.45

    Output:
    boxes - the location of boxes which outline detected objects
    scores - the confidence scores of each detection
    """
    boxes = []
    scores = []
    num_detections = detections.shape[1]
    for i in range(num_detections):
        x = detections[0, i]
        y = detections[1, i]
        width = detections[2, i]
        height = detections[3, i]
        confidence = detections[4, i]
        if confidence > confidence_threshold:
            x1 = int(x - width/ 2)
            y1 = int(y - height/ 2)
            x2 = int(x + width/ 2)
            y2 = int(y + height/ 2)
            boxes.append([x1, y1, x2 - x1, y2 - x1])
            scores.append(float(confidence))
    indices = cv.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold)
    final_boxes = []
    final_scores = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, width, height = boxes[i]
            final_boxes.append([x, y, x + width, y + height])
            final_scores.append(scores[i])
    return final_boxes, final_scores

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
        boxes, scores = process(detections)
        count = len(boxes)
        for (box, score) in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv.setWindowTitle(window_name, f"Car Detection | Count: {count}")
        cv.imshow(window_name, frame)

        if cv.waitKey(1) == 27: #ESC key to quit
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    counter()
