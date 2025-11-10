import cv2 as cv
import numpy as np


def run_model(img_name):
    """
    Runs best.onnx to make detections on a given image

    Parameters:
    img_name - name of image to detect

    Output:
    img - the open image in openCV
    detections - the detections that openCV made on the image
    height - height of the image
    width - width of the image
    """
    net = cv.dnn.readNetFromONNX("best.onnx")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    img = cv.imread(img_name)
    height, width = img.shape[:2]

    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    detections = detections.transpose()

    return img, detections, height, width


def counter(img_name="test.jpg"):
    """
    Counts how many cars are in the given image

    Parameters:
    img_name - the name of the image to make detections on. Defaults to test.jpg

    Output:
    count - the number of detected cars in the image
    """
    img, detections, height, width = run_model(img_name)
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
            cv.putText(
                img,
                label,
                (center_x - 20, center_y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    output = f"Car count: {count}"
    print(output)
    cv.imshow(output, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return count


if __name__ == "__main__":
    counter()
