# === predict.py ===
import os
from ultralytics import YOLO

if __name__ == "__main__":
    # Path to trained YOLOv8 model
    weights_path = "runs/exp_falcon/weights/best.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained model '{weights_path}' not found!")

    # Load YOLOv8 model
    model = YOLO(weights_path)

    # Automatically detect existing image folder
    possible_folders = [
        "falcon_dataset/test/images",
        "falcon_dataset/val/images",
        "falcon_dataset/train/images"
    ]

    source_folder = None
    for folder in possible_folders:
        if os.path.exists(folder) and len(os.listdir(folder)) > 0:
            source_folder = folder
            break

    if source_folder is None:
        raise FileNotFoundError("No images found for inference. Add images to test/val/train folders.")

    print(f"Running predictions on: {source_folder}")

    # Run predictions and save results
    results = model.predict(
        source=source_folder,
        save=True,
        project="runs",
        name="exp_falcon_preds"
    )

    print("✅ Predictions saved to 'runs/exp_falcon_preds/images'.")
