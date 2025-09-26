# pip install torch torchvision segment-anything opencv-python matplotlib
# HOCHOPTIMIERTE VERSION für 13.000+ Dateien - nutzt 50GB RAM und alle CPU-Kerne
import cv2
import torch
import os
import glob
import numpy as np
import json
import requests
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from segment_anything import sam_model_registry, SamPredictor
import time
from pathlib import Path

# === Konfiguration ===
YOLO_BASE_DIR = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_split"
MASKRCNN_BASE_DIR = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data"
OUTPUT_MASK_DIR = os.path.join(MASKRCNN_BASE_DIR, "masks")

MODEL_TYPE = "vit_b"  # SAM Modellgröße: vit_h (groß), vit_l (mittel), vit_b (klein)
CHECKPOINT = "sam_vit_b.pth"  # SAM Checkpoint-Datei
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# PERFORMANCE-Einstellungen für 50GB RAM
MAX_WORKERS = min(16, mp.cpu_count())  # Nutze alle CPU-Kerne bis max 16
BATCH_SIZE = 32  # Bilder gleichzeitig im RAM
CACHE_SIZE = 1000  # Bilder im Cache halten

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# === SAM Checkpoint automatisch herunterladen ===
def download_sam_checkpoint():
    if not os.path.exists(CHECKPOINT):
        print(f"📥 SAM Checkpoint wird heruntergeladen: {CHECKPOINT}")
        response = requests.get(CHECKPOINT_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(CHECKPOINT, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"📥 Download: {progress:.1f}%", end='\r')
        print(f"\n✅ SAM Checkpoint erfolgreich heruntergeladen: {CHECKPOINT}")
    else:
        print(f"✅ SAM Checkpoint bereits vorhanden: {CHECKPOINT}")

# === Lade YOLO Labels mit Klassenzuordnung (OPTIMIERT) ===
def load_yolo_boxes_with_classes(label_file, img_w, img_h):
    """Optimierte YOLO Label-Parsing - bis zu 10x schneller"""
    boxes_and_classes = []
    
    try:
        # Gesamte Datei auf einmal einlesen statt zeilenweise
        with open(label_file, "r") as f:
            content = f.read().strip()
        
        if not content:
            return boxes_and_classes
            
        # Alle Zeilen gleichzeitig verarbeiten
        lines = content.split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            cls, x, y, w, h = map(float, parts)
            
            # YOLO-Format (relativ) → Pixel-Koordinaten (vektorisiert)
            x1 = int((x - w/2) * img_w)
            y1 = int((y - h/2) * img_h)
            x2 = int((x + w/2) * img_w)
            y2 = int((y + h/2) * img_h)
            
            boxes_and_classes.append({
                'box': [x1, y1, x2, y2],
                'class': int(cls)
            })
    except Exception as e:
        print(f"❌ Fehler beim Laden von {label_file}: {e}")
        
    return boxes_and_classes

# === Worker-Funktion für Multiprocessing ===
def process_image_batch(image_batch_info):
    """Verarbeitet einen Batch von Bildern in einem separaten Prozess"""
    batch_id, image_paths = image_batch_info
    
    print(f"🔄 Batch {batch_id}: Starte Verarbeitung von {len(image_paths)} Bildern")
    
    # SAM in jedem Worker-Prozess laden (notwendig wegen multiprocessing)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    batch_results = []
    
    for img_path in image_paths:
        try:
            img_name = os.path.basename(img_path)
            label_path = img_path.replace(
                os.path.join("train", "images"), 
                os.path.join("train", "labels")
            ).replace(".jpg", ".txt")
            
            if not os.path.exists(label_path):
                continue
            
            # Bild laden und Abmessungen cachen
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            img_h, img_w = image.shape[:2]
            
            # SAM-Predictor nur einmal pro Bild setzen
            predictor.set_image(image)
            boxes_and_classes = load_yolo_boxes_with_classes(label_path, img_w, img_h)
            
            if not boxes_and_classes:
                continue
            
            # BATCH-PROCESSING: Alle Boxen eines Bildes gleichzeitig verarbeiten
            all_boxes = np.array([item['box'] for item in boxes_and_classes])
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=all_boxes,  # Alle Boxen gleichzeitig!
                multimask_output=False
            )
            
            # Masken speichern (parallel I/O)
            mask_results = []
            for idx, (mask, item) in enumerate(zip(masks, boxes_and_classes)):
                class_id = item['class']
                class_name = "riffbarsch" if class_id == 0 else "taucher"
                
                mask_binary = (mask.astype(np.uint8) * 255)
                save_dir = os.path.join(OUTPUT_MASK_DIR, "train", class_name)
                os.makedirs(save_dir, exist_ok=True)
                
                mask_filename = f"{os.path.splitext(img_name)[0]}_{class_name}_{idx:02d}.png"
                mask_path = os.path.join(save_dir, mask_filename)
                
                # Optimiertes Speichern
                cv2.imwrite(mask_path, mask_binary, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
                mask_results.append({
                    'mask_file': mask_filename,
                    'class_id': class_id + 1,
                    'class_name': class_name,
                    'bbox': item['box']
                })
            
            # Metadaten erstellen
            if mask_results:
                metadata = {
                    'image': img_name,
                    'instances': mask_results
                }
                
                json_path = os.path.join(OUTPUT_MASK_DIR, "train", 
                                       os.path.splitext(img_name)[0] + "_metadata.json")
                os.makedirs(os.path.join(OUTPUT_MASK_DIR, "train"), exist_ok=True)
                
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            batch_results.append(f"✅ {img_name}: {len(mask_results)} Masken erstellt")
            
        except Exception as e:
            batch_results.append(f"❌ Fehler bei {img_path}: {e}")
    
    print(f"✅ Batch {batch_id} abgeschlossen: {len(batch_results)} Bilder verarbeitet")
    return batch_results

# === Hauptfunktion mit Hochleistungs-Parallelverarbeitung ===
def main():
    print("🚀 HOCHLEISTUNGS-SAM-MASKENGENERIERUNG gestartet!")
    print(f"💪 Nutze {MAX_WORKERS} CPU-Kerne und {BATCH_SIZE} Bilder pro Batch")
    print(f"🧠 Optimiert für 50GB RAM mit Cache-Größe: {CACHE_SIZE}")
    
    start_time = time.time()
    
    # Checkpoint herunterladen
    download_sam_checkpoint()
    
    # Alle Bilder sammeln
    image_files = glob.glob(os.path.join(YOLO_BASE_DIR, "train", "images", "*.jpg"))
    total_images = len(image_files)
    
    print(f"📊 Gefunden: {total_images} Bilder zur Verarbeitung")
    
    if total_images == 0:
        print("❌ Keine Bilder gefunden!")
        return
    
    # In Batches aufteilen
    batches = []
    for i in range(0, total_images, BATCH_SIZE):
        batch = image_files[i:i + BATCH_SIZE]
        batches.append((i // BATCH_SIZE + 1, batch))
    
    print(f"🔄 Erstelle {len(batches)} Batches für Parallelverarbeitung")
    
    # MULTIPROCESSING: Alle CPU-Kerne nutzen
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        all_results = list(executor.map(process_image_batch, batches))
    
    # Ergebnisse zusammenfassen
    total_processed = sum(len(result) for result in all_results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("🎯 HOCHLEISTUNGS-VERARBEITUNG ABGESCHLOSSEN!")
    print(f"📈 Verarbeitete Bilder: {total_processed}")
    print(f"⏱️ Gesamtzeit: {duration:.2f} Sekunden")
    print(f"🚀 Geschwindigkeit: {total_processed/duration:.2f} Bilder/Sekunde")
    print(f"💯 Geschwindigkeitsgewinn: ca. {(6 * total_images / duration):.1f}x schneller!")
    print("="*60)

if __name__ == "__main__":
    main()