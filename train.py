# === train.py ===
from ultralytics import YOLO
import os

if __name__ == "__main__":
    # Path to dataset YAML
    data_yaml = "falcon.yaml"
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config '{data_yaml}' not found!")

    # Initialize YOLOv8 model (smallest for faster training, can change to yolov8s/m/l)
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data=data_yaml,
        epochs=50,          # change epochs if needed
        imgsz=640,          # image size
        batch=16,           # batch size
        project="runs",     # save outputs here
        name="exp_falcon",  # experiment name
        exist_ok=True       # overwrite if folder exists
    )

    print("✅ Training complete! Check 'runs/exp_falcon' for outputs.")
