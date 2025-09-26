from ultralytics import YOLO

# Modell laden (vortrainiert)
model = YOLO("yolov8m.pt")  # 'n', 's', 'm' je nach GPU; 'm' ist guter Start

results = model.train(
    data="yolo_reef.yaml",
    imgsz=1024,          # bei Tiling 1024; sonst 1280 testen
    epochs=200,
    batch=8,             # an VRAM anpassen
    optimizer="adamw",
    lr0=1e-3,
    cos_lr=True,
    patience=50,         # Early Stopping
    mosaic=0.7,          # moderat
    mixup=0.1,
    hsv_h=0.02, hsv_s=0.6, hsv_v=0.6,
    degrees=5, translate=0.05, scale=0.2, shear=0.0,
    fliplr=0.5,
    box=7.5, cls=0.5, dfl=1.5,  # Loss-Gewichte (feintunen)
    workers=8
)
