# detect.py — inference script using Ultralytics YOLO API
from ultralytics import YOLO
import argparse
import cv2


def detect(weights='runs/train/exp/weights/best.pt', source=0, conf=0.25, save=True, show=True):
model = YOLO(weights)
results = model.predict(source=source, conf=conf, save=False)


# If source is an image or video file, results contains boxes per frame.
# We'll visualize using OpenCV for a single image or camera.
for r in results:
im = r.orig_img
boxes = r.boxes
for box in boxes:
xyxy = box.xyxy[0].cpu().numpy().astype(int)
conf_score = float(box.conf[0].cpu().numpy())
cls = int(box.cls[0].cpu().numpy())
label = f"{model.names[cls]} {conf_score:.2f}"
cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255,255,255), 2)
cv2.putText(im, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
if show:
cv2.imshow('det', im)
if cv2.waitKey(0) & 0xFF == ord('q'):
break
if save:
# save first result image
if len(results) and hasattr(results[0], 'orig_img'):
cv2.imwrite('runs/detect/result.jpg', results[0].orig_img)


if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='runs/train/exp/weights/best.pt')
parser.add_argument('--source', default=0)
parser.add_argument('--conf', type=float, default=0.25)
parser.add_argument('--save', action='store_true')
parser.add_argument('--show', action='store_true')
args = parser.parse_args()
detect(weights=args.weights, source=args.source, conf=args.conf, save=args.save, show=args.show)
