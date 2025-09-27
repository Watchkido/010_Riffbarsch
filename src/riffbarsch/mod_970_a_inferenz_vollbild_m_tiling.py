import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

def infer_tiled(img_path, model, tile=1024, overlap=0.2, conf=0.25, iou=0.6):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    stride = int(tile * (1 - overlap))
    dets = []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1, y1 = min(x0+tile, w), min(y0+tile, h)
            if x1 - x0 < tile or y1 - y0 < tile:
                continue
            crop = img.crop((x0, y0, x1, y1))
            res = model.predict(source=crop, imgsz=1024, conf=conf, iou=iou, verbose=False)[0]
            for b in res.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                score = float(b.conf.cpu().numpy())
                cls_id = int(b.cls.cpu().numpy())
                # Rückverschiebung
                x_min, y_min, x_max, y_max = xyxy
                dets.append([x_min+x0, y_min+y0, x_max+x0, y_max+y0, score, cls_id])

    # NMS über alle Kacheln
    boxes = torch.tensor([d[:4] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in dets], dtype=torch.float32)
    keep = torch.ops.torchvision.nms(boxes, scores, iou)
    final = [dets[i] for i in keep]
    return final

model = YOLO(r"E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n\riffbarsch_taucher_run\weights\best.pt")
final_dets = infer_tiled("some_highres.jpg", model, tile=1024, overlap=0.2, conf=0.25, iou=0.6)
print(final_dets)  # [x1, y1, x2, y2, score, class]
