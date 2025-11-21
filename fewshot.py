from ultralytics import YOLO

# Load pretrained OBB model
model = YOLO("yolo11s-obb.pt")  

# Fine-tune Task 1
results = model.train(
    data="dior_task1/dior_task1.yaml",  # YAML dataset Task 1
    epochs=50,                          # số epoch
    imgsz=1024,                          # kích thước ảnh
    batch=8,                             # batch size
    device=4,                            # GPU index
    name="dior_task1_fewshot"
)

metrics_test = model.val(
    data="dior_task1/dior_task1.yaml",
    batch=8,
    device=4,
    split="test",
    name="dior_task1_fewshot_val"
)
print("Final test metrics:")
print(metrics_test)
