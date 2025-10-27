import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

dataset_dir = "dataset"
output_dir = "yolo_dataset"
for split in ['train', 'val']:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

image_files = os.listdir(f"{dataset_dir}/images")
train_imgs, val_imgs = train_test_split(image_files, test_size=0.2, random_state=42)
for split_name, img_list in [('train', train_imgs), ('val', val_imgs)]:
    for img_file in img_list:
        img_name = os.path.splitext(img_file)[0]
        shutil.copy(
            f"{dataset_dir}/images/{img_file}",
            f"{output_dir}/images/{split_name}/{img_file}"
        )

        shutil.copy(
            f"{dataset_dir}/labels/{img_name}.txt",
            f"{output_dir}/labels/{split_name}/{img_name}.txt"
        )

with open(f"{dataset_dir}/classes.txt") as file:
    classes = [line.strip() for line in file]

yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""

with open(f"{output_dir}/data.yaml", 'w') as file:
    file.write(yaml_content)

model = YOLO('yolov8n-obb.pt')
model.train(
    data = 'yolo_dataset/data.yaml',
    epochs = 100,
    imgsz=640
)
model.export(format='onnx')