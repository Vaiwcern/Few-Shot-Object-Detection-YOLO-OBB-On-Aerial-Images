# import os

# # Thư mục gốc labels
# label_root = "dior_task1/labels"
# subfolders = ["train", "val", "test"]

# # Mapping class gốc -> class mới 0-3
# # 5: Chimney, 6: Dam, 14: Stadium, 19: Wind mill
# class_map = {
#     5: 0,   # Chimney
#     6: 1,   # Dam
#     14: 2,  # Stadium
#     19: 3   # Wind mill
# }

# for sub in subfolders:
#     folder_path = os.path.join(label_root, sub)
#     for fname in os.listdir(folder_path):
#         if not fname.endswith(".txt"):
#             continue
#         file_path = os.path.join(folder_path, fname)
#         new_lines = []
#         with open(file_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 9:
#                     continue
#                 cls = int(parts[0])
#                 if cls in class_map:
#                     parts[0] = str(class_map[cls])
#                     new_lines.append(" ".join(parts))
#         # Ghi đè file label
#         with open(file_path, "w") as f:
#             f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
# print("Đã map class và xóa các class không cần thiết!")

import os

# Thư mục gốc labels của task 2
label_root = "dior_task3/context_adapt/labels"
subfolders = ["train", "val"]

# 4 class few-shot trong task 1 (không được xuất hiện trong context)
few_shot_classes = [5, 6, 14, 19]  # Chimney, Dam, Stadium, Wind mill

# Mapping class gốc -> class mới 0-(nc-1)
# Chỉ giữ các class không thuộc few-shot
# Lấy các class còn lại trong DIOR gốc: 0-19 trừ few_shot_classes
remaining_classes = [i for i in range(20) if i not in few_shot_classes]
class_map = {cls: idx for idx, cls in enumerate(remaining_classes)}

for sub in subfolders:
    folder_path = os.path.join(label_root, sub)
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        file_path = os.path.join(folder_path, fname)
        new_lines = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls = int(parts[0])
                if cls in class_map:
                    parts[0] = str(class_map[cls])
                    new_lines.append(" ".join(parts))
        # Ghi đè file label
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

print("Đã remap class cho task 2 và xóa các class few-shot không cần thiết!")
