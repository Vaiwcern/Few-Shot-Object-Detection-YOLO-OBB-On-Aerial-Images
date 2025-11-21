import os
import random
import shutil
from collections import defaultdict

# ---------------- CONFIG ----------------
dior_root = "dior_converted"  # folder gốc DIOR
images_folder = os.path.join(dior_root, "images", "train")
labels_folder = os.path.join(dior_root, "labels", "train")

output_root = "dior_task3/context_adapt"
num_per_class = 600  # số ảnh/context per class
val_total = 30       # tổng số ảnh val

fewshot_classes = ["Chimney", "Dam", "Stadium", "Wind mill"]
all_classes = ["Airplane","Airport","Baseball field","Basketball court","Bridge",
               "Chimney","Dam","Expressway service area","Expressway toll station",
               "Golf course","Ground track field","Harbor","Overpass","Ship",
               "Stadium","Storage tank","Tennis court","Train station","Vehicle","Wind mill"]

# Lấy các class còn lại
context_classes = [c for c in all_classes if c not in fewshot_classes]

random.seed(42)

# ---------------- Tạo folder output ----------------
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)

# ---------------- Đọc label và đếm object ----------------
class_to_imgs = defaultdict(list)
for lbl_file in os.listdir(labels_folder):
    lbl_path = os.path.join(labels_folder, lbl_file)
    with open(lbl_path, "r") as f:
        lines = f.readlines()
        # check xem ảnh có object few-shot không
        skip = False
        counts = defaultdict(int)
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            id_to_name = {
                0:"Airplane", 1:"Airport", 2:"Baseball field", 3:"Basketball court",
                4:"Bridge", 5:"Chimney", 6:"Dam", 7:"Expressway service area",
                8:"Expressway toll station", 9:"Golf course", 10:"Ground track field",
                11:"Harbor", 12:"Overpass", 13:"Ship", 14:"Stadium", 15:"Storage tank",
                16:"Tennis court", 17:"Train station", 18:"Vehicle", 19:"Wind mill"
            }
            cls_name = id_to_name[cls_id]
            if cls_name in fewshot_classes:
                skip = True
                break
            if cls_name in context_classes:
                counts[cls_name] += 1
        if skip:
            continue
        for cls_name, cnt in counts.items():
            if cnt > 0:
                class_to_imgs[cls_name].append((lbl_file, cnt))

# ---------------- Chọn ảnh nhiều object trước ----------------
selected_imgs = []
for cls_name in context_classes:
    imgs = sorted(class_to_imgs[cls_name], key=lambda x: -x[1])
    chosen = imgs[:num_per_class]
    selected_imgs.extend([f for f,_ in chosen])

# ---------------- Shuffle và chia train/val ----------------
random.shuffle(selected_imgs)
val_imgs = selected_imgs[:val_total]
train_imgs = selected_imgs[val_total:]

# ---------------- Copy ảnh + label ----------------
for split, split_imgs in zip(["train","val"], [train_imgs, val_imgs]):
    for lbl_file in split_imgs:
        base_name = os.path.splitext(lbl_file)[0]
        img_file = base_name + ".jpg"

        src_img = os.path.join(images_folder, img_file)
        src_lbl = os.path.join(labels_folder, lbl_file)

        dst_img = os.path.join(output_root, "images", split, img_file)
        dst_lbl = os.path.join(output_root, "labels", split, lbl_file)

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

print("✅ Context adaptation subset created at", output_root)
