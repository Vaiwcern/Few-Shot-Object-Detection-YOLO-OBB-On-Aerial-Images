import os
import random
from PIL import Image, ImageDraw

# --- Config ---
dior_converted = "dior_converted"
split = "train"  # bạn có thể đổi sang val/test
images_folder = os.path.join(dior_converted, "images", split)
labels_folder = os.path.join(dior_converted, "labels", split)

# --- Chọn random image ---
img_file = random.choice(os.listdir(images_folder))
img_path = os.path.join(images_folder, img_file)
lbl_path = os.path.join(labels_folder, os.path.splitext(img_file)[0] + ".txt")

# --- Mở ảnh ---
img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)
width, height = img.size

# --- Đọc label OBB và vẽ box ---
with open(lbl_path, "r") as f:
    for line in f.readlines():
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))  # x1 y1 x2 y2 x3 y3 x4 y4 normalized
        # chuyển sang pixel
        coords_px = [(coords[i]*width, coords[i+1]*height) for i in range(0,8,2)]
        # vẽ polygon
        draw.line(coords_px + [coords_px[0]], width=2, fill="red")

# --- Lưu ảnh kiểm tra ---
img.save("test.jpg")
print("✅ Lưu ảnh kiểm tra tại test.jpg với bounding box vẽ đỏ")
