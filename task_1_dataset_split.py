import os
import random
import shutil
from collections import defaultdict

# ---------------- CONFIG ----------------
dior_root = "dior_converted"  # folder gốc DIOR
images_folder = os.path.join(dior_root, "images", "train")  # chỉ lấy từ train
labels_folder = os.path.join(dior_root, "labels", "train")

output_root = "dior_task1"
classes = ["Chimney", "Dam", "Stadium", "Wind mill"]  # 4 class
num_train = 15
num_val = 5
num_test = 5

random.seed(42)

# ---------------- Tạo folder output ----------------
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)

# ---------------- Đọc label và đánh giá số object ----------------
class_to_imgs = defaultdict(list)

for lbl_file in os.listdir(labels_folder):
    lbl_path = os.path.join(labels_folder, lbl_file)
    with open(lbl_path, "r") as f:
        lines = f.readlines()
        counts = defaultdict(int)
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            # mapping id → class name theo DIOR
            id_to_name = {
                5: "Chimney",
                6: "Dam",
                14: "Stadium",
                19: "Wind mill"
            }
            if cls_id in id_to_name:
                counts[id_to_name[cls_id]] += 1
        for cls_name, cnt in counts.items():
            if cnt > 0:
                class_to_imgs[cls_name].append((lbl_file, cnt))

# ---------------- Chọn ảnh nhiều object trước ----------------
selected_imgs = defaultdict(list)

for cls_name in classes:
    # sort giảm dần theo số object
    imgs = sorted(class_to_imgs[cls_name], key=lambda x: -x[1])
    chosen = imgs[:num_train + num_val + num_test]
    selected_imgs[cls_name] = [f for f, _ in chosen]

# ---------------- Copy ảnh + label ----------------
for cls_name in classes:
    imgs = selected_imgs[cls_name]
    random.shuffle(imgs)
    train_imgs = imgs[:num_train]
    val_imgs = imgs[num_train:num_train+num_val]
    test_imgs = imgs[num_train+num_val:num_train+num_val+num_test]

    for split, split_imgs in zip(["train","val","test"], [train_imgs, val_imgs, test_imgs]):
        for lbl_file in split_imgs:
            base_name = os.path.splitext(lbl_file)[0]
            img_file = base_name + ".jpg"

            src_img = os.path.join(images_folder, img_file)
            src_lbl = os.path.join(labels_folder, lbl_file)

            dst_img = os.path.join(output_root, "images", split, img_file)
            dst_lbl = os.path.join(output_root, "labels", split, lbl_file)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

print("✅ Task 1 dataset created at", output_root)
