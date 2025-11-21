import os
import glob
import shutil
import numpy as np
import yaml

# --- Config ---
src_root = "dior"           # folder gốc dataset YOLO AABB
dst_root = "dior_converted" # folder output YOLO-OBB
splits = ["train", "val", "test"]  # các split bạn muốn convert
aabb_labels_folder = os.path.join(src_root, "labels")
images_folder = os.path.join(src_root, "images")

# 20 class DIOR
names = {
    0: "Airplane",
    1: "Airport",
    2: "Baseball field",
    3: "Basketball court",
    4: "Bridge",
    5: "Chimney",
    6: "Dam",
    7: "Expressway service area",
    8: "Expressway toll station",
    9: "Golf course",
    10: "Ground track field",
    11: "Harbor",
    12: "Overpass",
    13: "Ship",
    14: "Stadium",
    15: "Storage tank",
    16: "Tennis court",
    17: "Train station",
    18: "Vehicle",
    19: "Wind mill"
}

# --- Hàm convert XYWH -> 4 points OBB (normalized) ---
def aabb_to_obb(xc, yc, w, h):
    """
    Input: xywh normalized
    Output: 8 numbers = x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
    Order: top-left, top-right, bottom-right, bottom-left
    """
    x1 = xc - w/2
    y1 = yc - h/2
    x2 = xc + w/2
    y2 = yc - h/2
    x3 = xc + w/2
    y3 = yc + h/2
    x4 = xc - w/2
    y4 = yc + h/2
    return [x1, y1, x2, y2, x3, y3, x4, y4]

# --- Tạo folder output ---
for split in splits:
    os.makedirs(os.path.join(dst_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "labels", split), exist_ok=True)

# --- Đọc các ImageSets *.txt để lấy danh sách ảnh theo split ---
for split in splits:
    txt_file = os.path.join(src_root, "ImageSets", f"{split}.txt")
    with open(txt_file, "r") as f:
        image_names = [line.strip() for line in f.readlines()]

    for name in image_names:
        # Copy ảnh
        src_img = os.path.join(images_folder, f"{name}.jpg")
        dst_img = os.path.join(dst_root, "images", split, f"{name}.jpg")
        shutil.copy2(src_img, dst_img)

        # Đọc label AABB
        src_lbl = os.path.join(aabb_labels_folder, f"{name}.txt")
        dst_lbl = os.path.join(dst_root, "labels", split, f"{name}.txt")
        if not os.path.exists(src_lbl):
            continue

        with open(src_lbl, "r") as f:
            lines = f.readlines()

        with open(dst_lbl, "w") as f:
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                pts = aabb_to_obb(xc, yc, w, h)
                line_out = f"{cls} " + " ".join(f"{p:.6f}" for p in pts)
                f.write(line_out + "\n")

# --- Tạo file YAML cho YOLO ---
yaml_content = {
    "path": dst_root,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": names
}

yaml_path = os.path.join(dst_root, "diordata.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f)

print("✅ Hoàn thành convert YOLO AABB -> YOLO OBB tại folder:", dst_root)
