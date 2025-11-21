from ultralytics import YOLO

# Load pretrained OBB model
model = YOLO("yolo11s-obb.pt") 

# Fine-tune Task 1
results = model.train(
    data="dior_task2/context_adapt/dior_task2.yaml",  # YAML dataset Task 1
    epochs=50,                          # số epoch
    imgsz=1024,                          # kích thước ảnh
    batch=8,                             # batch size
    device=7,                            # GPU index
    freeze=10,
    name="dior_task2_finetune"
)

# Evaluate cuối cùng trên test set
metrics_test = model.val(
    data="dior_task2/context_adapt/dior_task2.yaml",
    batch=8,
    device=7,
    split="test", 
    name="dior_task2_finetune_val"
)

print("Final test metrics:")
print(metrics_test)
