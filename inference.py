from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

# Load your trained model
model = YOLO("best.pt")   # or roboflow_yolo_best.pt

# Image URL
url = "https://www.tripcheck.com/roadcams/cams/I-5atArndtRd_pid798.jpg?0.4944914436602593"

# Download image
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Run inference
results = model(img)

# Show results
results[0].show()

# Optional: save annotated image
results[0].save(filename="prediction.jpg")
