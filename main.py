from ultralytics import YOLO
import pandas as pd
import cv2
import os

# 1️⃣ Load pretrained model YOLO11n-OBB
model = YOLO("yolov8n-obb.pt")

# 2️⃣ Input image
img_path = "/raid/ltnghia02/tnh/P0492.jpg"
img_name = os.path.basename(img_path)

# 3️⃣ Predict
results = model(img_path, conf=0.25)  # hoặc model("img_path")

# 4️⃣ Save annotated image
annotated_img = results[0].plot()
out_img_path = img_name.replace(".jpg", "_result.jpg")
cv2.imwrite(out_img_path, annotated_img)

# 5️⃣ Extract OBB results and save CSV
boxes_data = []

r = results[0]  # single image
if r.obb is not None and len(r.obb.xywhr) > 0:
    xywhr = r.obb.xywhr.cpu().numpy()        # Nx5: cx, cy, w, h, angle (radians)
    xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()  # Nx8: 4 corners polygon
    cls = r.obb.cls.cpu().numpy()            # class indices
    conf = r.obb.conf.cpu().numpy()          # confidence scores
    names = [r.names[int(c)] for c in cls]

    for b, c, f, n in zip(xywhr, cls, conf, names):
        boxes_data.append([img_name, int(c), n, float(f), *b])  # cx,cy,w,h,angle

# 6️⃣ Save CSV
csv_path = img_name.replace(".jpg", "_result.csv")
df = pd.DataFrame(boxes_data, columns=["image","class_id","class_name","conf","center_x","center_y","w","h","angle"])
df.to_csv(csv_path, index=False)

print(f"Done. Annotated image saved to {out_img_path}, CSV saved to {csv_path}")
