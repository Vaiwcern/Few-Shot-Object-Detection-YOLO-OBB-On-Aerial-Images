from ultralytics import YOLO

# Load pretrained OBB model
model = YOLO("yolo11s-obb-finetune-10k-freeze.pt")  # hoặc đường dẫn tới pretrained


# Fine-tune Task 1
results = model.train(
    data="dior_task1/dior_task1.yaml",  # YAML dataset Task 1
    epochs=50,                          # số epoch
    imgsz=1024,                          # kích thước ảnh
    batch=8,                             # batch size
    device=7,                            # GPU index
    freeze=10,
    name="dior_task1_tfa"
)

# Evaluate cuối cùng trên test set
metrics_test = model.val(
    data="dior_task1/dior_task1.yaml",
    batch=8,
    device=7,
    split="test",  # split test để đánh giá cuối cùng
    name="dior_task1_tfa_val"
)
print("Final test metrics:")
print(metrics_test)
