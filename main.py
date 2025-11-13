# main.py
# Falcon YOLOv8 Project - End-to-End Pipeline
# Includes: Setup, Download, Prepare Dataset, Train, Predict, Package

import os
import zipfile
import shutil
import yaml
import random
import cv2
import matplotlib.pyplot as plt
import gdown
from ultralytics import YOLO

# =====================
# Step 1: Setup & Install
# =====================
print("🚀 Falcon YOLOv8 Project Starting...")

# Google Drive Falcon dataset link
file_id = "1zOvorLRxaxEJxqY8YyMmP3KI1nPXX70Z"
url = f"https://drive.google.com/uc?id={file_id}"
output = "falcon_dataset.zip"

# Download if not already present
if not os.path.exists(output):
    print("📥 Downloading dataset...")
    gdown.download(url, output, quiet=False)

# Unzip dataset
if not os.path.exists("falcon_dataset"):
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("falcon_dataset")
print("✅ Dataset ready!")

# =====================
# Step 2: Explore Dataset
# =====================
def show_random_examples(img_dir, n=4):
    if not os.path.exists(img_dir):
        print(f"⚠️ Folder {img_dir} does not exist")
        return
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    samples = random.sample(imgs, min(n, len(imgs)))
    plt.figure(figsize=(10,10))
    for i, img_name in enumerate(samples):
        img = cv2.imread(os.path.join(img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2,2,i+1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

train_img_dir = "falcon_dataset/train/images"
val_img_dir = "falcon_dataset/val/images"

print("📊 Showing sample images...")
show_random_examples(train_img_dir, n=4)

# =====================
# Step 3: Prepare falcon.yaml
# =====================
dataset_yaml = {
    "train": train_img_dir,
    "val": val_img_dir,
    "test": val_img_dir,
    "nc": 7,
    "names": [
        "Oxygen Tank", "Nitrogen Tank", "First Aid Box",
        "Fire Alarm", "Safety Switch Panel", "Emergency Phone",
        "Fire Extinguisher"
    ]
}

with open("falcon.yaml", "w") as f:
    yaml.dump(dataset_yaml, f)

print("✅ falcon.yaml created")

# =====================
# Step 4: Train Model
# =====================
print("🚀 Training YOLOv8 model...")
model = YOLO("yolov8n.pt")

model.train(
    data="falcon.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs",
    name="exp_falcon"
)

print("✅ Training finished!")

# =====================
# Step 5: Inference / Prediction
# =====================
print("🔮 Running inference...")
out_dir = "runs/exp_falcon/predictions"
os.makedirs(out_dir, exist_ok=True)

results = model.predict(
    source=val_img_dir,
    save=True,
    project="runs",
    name="exp_falcon_preds"
)

pred_dir = os.path.join("runs", "exp_falcon_preds", "images")
if os.path.exists(pred_dir):
    pred_imgs = os.listdir(pred_dir)
    for img_name in pred_imgs[:4]:
        img = cv2.imread(os.path.join(pred_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
else:
    print("⚠️ No predictions found")

# =====================
# Step 6: Package for Submission
# =====================
print("📦 Packaging project...")

submission_folder = "Final_Submission"
if os.path.exists(submission_folder):
    shutil.rmtree(submission_folder)

os.makedirs(submission_folder)

# Copy files
shutil.copy("falcon.yaml", submission_folder)
shutil.copytree("runs", os.path.join(submission_folder, "runs"))

# Zip it
shutil.make_archive("Final_Submission", 'zip', submission_folder)

print("✅ Final_Submission.zip created! Ready to upload 🚀")
