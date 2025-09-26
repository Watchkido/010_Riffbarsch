# pip install torch torchvision segment-anything opencv-python matplotlib
# https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
import cv2
import torch
import os
import glob
import numpy as np
import json
import requests
from segment_anything import sam_model_registry, SamPredictor

# === Konfiguration ===
YOLO_BASE_DIR = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_split"
MASKRCNN_BASE_DIR = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data"
OUTPUT_MASK_DIR = os.path.join(MASKRCNN_BASE_DIR, "masks")

MODEL_TYPE = "vit_b"  # SAM Modellgröße: vit_h (groß), vit_l (mittel), vit_b (klein)
CHECKPOINT = "sam_vit_b.pth"  # SAM Checkpoint-Datei herunterladen von Meta AI
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# === SAM Checkpoint automatisch herunterladen falls nicht vorhanden ===
def download_sam_checkpoint():
    if not os.path.exists(CHECKPOINT):
        print(f"📥 SAM Checkpoint wird heruntergeladen: {CHECKPOINT}")
        response = requests.get(CHECKPOINT_URL, stream=True)
        response.raise_for_status()
        
        with open(CHECKPOINT, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ SAM Checkpoint erfolgreich heruntergeladen: {CHECKPOINT}")
    else:
        print(f"✅ SAM Checkpoint bereits vorhanden: {CHECKPOINT}")

# Checkpoint herunterladen
download_sam_checkpoint()

# === Lade SAM Modell ===
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

# === Hilfsfunktion: Lade YOLO Labels mit Klassenzuordnung ===
def load_yolo_boxes_with_classes(label_file, img_w, img_h):
    boxes_and_classes = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            # YOLO-Format (relativ) → Pixel-Koordinaten
            x1 = int((x - w/2) * img_w)
            y1 = int((y - h/2) * img_h)
            x2 = int((x + w/2) * img_w)
            y2 = int((y + h/2) * img_h)
            boxes_and_classes.append({
                'box': [x1, y1, x2, y2],
                'class': int(cls)  # 0=riffbarsch, 1=taucher
            })
    return boxes_and_classes

# === Hauptschleife: Alle Bilder im YOLO-Dataset verarbeiten ===
image_files = glob.glob(os.path.join(YOLO_BASE_DIR, "train", "images", "*.jpg"))

for img_path in image_files:
    img_name = os.path.basename(img_path)
    # YOLO Labels befinden sich in einem parallelen "labels" Ordner
    label_path = img_path.replace(os.path.join("train", "images"), os.path.join("train", "labels")).replace(".jpg", ".txt")

    if not os.path.exists(label_path):
        print(f"⚠️ Keine Labels für {img_name}, übersprungen.")
        continue

    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]

    predictor.set_image(image)
    boxes_and_classes = load_yolo_boxes_with_classes(label_path, img_w, img_h)

    # Speichere jede Instanz als separate Maske
    for idx, item in enumerate(boxes_and_classes):
        box = np.array(item['box'])
        class_id = item['class']
        
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],  # SAM erwartet Array
            multimask_output=False
        )
        
        mask = masks[0].astype(np.uint8) * 255
        class_name = "riffbarsch" if class_id == 0 else "taucher"
        
        # Speichern im Mask R-CNN Verzeichnis nach Klassen unterteilt
        class_name = "riffbarsch" if class_id == 0 else "taucher"
        save_dir = os.path.join(OUTPUT_MASK_DIR, "train", class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        mask_filename = f"{os.path.splitext(img_name)[0]}_{class_name}_{idx:02d}.png"
        mask_path = os.path.join(save_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        print(f"✅ Maske gespeichert: {mask_path}")

    # Erstelle zusätzlich eine JSON-Datei mit Metadaten für das Training
    if len(boxes_and_classes) > 0:
        metadata = {
            'image': img_name,
            'instances': []
        }
        
        for idx, item in enumerate(boxes_and_classes):
            class_name = "riffbarsch" if item['class'] == 0 else "taucher"
            mask_filename = f"{os.path.splitext(img_name)[0]}_{class_name}_{idx:02d}.png"
            metadata['instances'].append({
                'mask_file': mask_filename,
                'class_id': item['class'] + 1,  # +1 weil 0=Hintergrund in Mask R-CNN
                'class_name': class_name,
                'bbox': item['box']
            })
        
        # Speichere Metadaten im Mask-Hauptverzeichnis (alle zusammen)
        json_path = os.path.join(OUTPUT_MASK_DIR, "train", os.path.splitext(img_name)[0] + "_metadata.json")
        os.makedirs(os.path.join(OUTPUT_MASK_DIR, "train"), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    if len(boxes_and_classes) == 0:
        print(f"⚠️ Keine Objekte in {img_name}")

print("🎯 Alle Masken und Metadaten wurden erfolgreich erstellt!")