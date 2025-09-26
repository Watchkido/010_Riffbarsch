#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 MASK R-CNN TRAINING - RAM TURBO VERSION 
Hochgeschwindigkeits-Training mit vollständigem RAM-Cache für 128GB RAM
Alle Bilder und Masken werden vorab in den Speicher geladen.

Erstellt: 2025-09-25
Autor: Frank Albrecht - Der RAM-Optimierungsguru
"""

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
from pathlib import Path
import time
import gc
from typing import Dict, List, Tuple, Any
import psutil

# --- TURBO PARAMETER für 128GB RAM ---
BATCH_SIZE = 8        # Deutlich höher dank RAM-Cache
NUM_EPOCHS = 10       # Zurück auf ursprüngliche Anzahl
NUM_CLASSES = 3       # Hintergrund + riffbarsch + taucher (1, 2)
RAM_CACHE_SIZE_GB = 60  # Nutze 60GB für Dataset-Cache
PREFETCH_WORKERS = 8   # Paralleles Vorladen

# Pfade
DATA_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data")
MASKS_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data\masks")
OUTPUT_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\models\maskrcnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")  # Vorerst CPU, später optional GPU
print(f"🖥️ Training auf: {DEVICE}")

def get_ram_usage():
    """Aktuelle RAM-Nutzung in GB"""
    return psutil.virtual_memory().used / (1024**3)

class RAMTurboMaskDataset(Dataset):
    """
    🚀 RAM-TURBO Dataset Klasse
    Lädt ALLE Daten beim Initialisieren in den RAM für maximale Geschwindigkeit
    """
    def __init__(self, root, masks_dir, transforms=None):
        print(f"🚀 INITIALISIERE RAM-TURBO DATASET...")
        start_time = time.time()
        
        self.root = Path(root)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        
        # RAM-Cache für alle Daten
        self.image_cache: Dict[str, torch.Tensor] = {}
        self.mask_cache: Dict[str, torch.Tensor] = {}
        self.metadata_cache: List[Dict] = []
        
        # Finde alle Metadaten
        metadata_files = list(self.masks_dir.glob("train/*_metadata.json"))
        print(f"📊 Gefunden: {len(metadata_files)} Metadaten-Dateien")
        
        # SCHRITT 1: Lade alle Metadaten in RAM
        print(f"🔄 Lade Metadaten in RAM...")
        for meta_file in metadata_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                self.metadata_cache.append(metadata)
        
        print(f"✅ {len(self.metadata_cache)} Metadaten geladen")
        
        # SCHRITT 2: Lade alle Bilder in RAM
        print(f"🔄 Lade alle Bilder in RAM-Cache...")
        unique_images = set()
        for metadata in self.metadata_cache:
            unique_images.add(metadata['image'])
        
        yolo_img_path_base = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_split\train\images")
        
        for i, img_name in enumerate(unique_images):
            if i % 50 == 0:
                ram_gb = get_ram_usage()
                print(f"   Bild {i+1}/{len(unique_images)} - RAM: {ram_gb:.1f}GB")
            
            img_path = yolo_img_path_base / img_name
            if img_path.exists():
                # Lade und konvertiere Bild zu Tensor
                img = Image.open(img_path).convert("RGB")
                img_tensor = T.ToTensor()(img)
                self.image_cache[img_name] = img_tensor
            else:
                print(f"⚠️ Bild nicht gefunden: {img_name}")
        
        print(f"✅ {len(self.image_cache)} Bilder im RAM-Cache")
        
        # SCHRITT 3: Lade alle Masken in RAM
        print(f"🔄 Lade alle Masken in RAM-Cache...")
        mask_count = 0
        for metadata in self.metadata_cache:
            for instance in metadata['instances']:
                mask_filename = instance['mask_file']
                class_name = instance['class_name']
                mask_path = self.masks_dir / "train" / class_name / mask_filename
                
                if mask_path.exists() and mask_filename not in self.mask_cache:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_array = np.array(mask_img) > 0  # Binäre Maske
                    mask_tensor = torch.as_tensor(mask_array, dtype=torch.uint8)
                    self.mask_cache[mask_filename] = mask_tensor
                    mask_count += 1
                    
                    if mask_count % 100 == 0:
                        ram_gb = get_ram_usage()
                        print(f"   Maske {mask_count} - RAM: {ram_gb:.1f}GB")
        
        print(f"✅ {len(self.mask_cache)} Masken im RAM-Cache")
        
        load_time = time.time() - start_time
        final_ram = get_ram_usage()
        print(f"🎉 RAM-TURBO DATASET BEREIT!")
        print(f"⏱️ Ladezeit: {load_time:.1f} Sekunden")
        print(f"💾 RAM-Nutzung: {final_ram:.1f}GB")
        print(f"📊 Dataset-Größe: {len(self.metadata_cache)} Samples")

    def __getitem__(self, idx):
        """
        🚀 ULTRA-SCHNELLER Datenzugriff aus RAM-Cache
        """
        metadata = self.metadata_cache[idx]
        img_name = metadata['image']
        
        # Lade Bild aus RAM-Cache
        if img_name in self.image_cache:
            img = self.image_cache[img_name].clone()  # Clone für Sicherheit
            _, img_height, img_width = img.shape
        else:
            raise FileNotFoundError(f"Bild nicht im Cache: {img_name}")
        
        # Erstelle Listen für alle Instanzen
        boxes = []
        labels = []
        masks = []
        
        for instance in metadata['instances']:
            # Bounding Box
            bbox = instance['bbox']  # [x1, y1, x2, y2]
            boxes.append(bbox)
            
            # Label
            labels.append(instance['class_id'])
            
            # Lade Maske aus RAM-Cache
            mask_filename = instance['mask_file']
            if mask_filename in self.mask_cache:
                mask = self.mask_cache[mask_filename].clone()
                masks.append(mask)
            else:
                # Fallback: Dummy-Maske
                dummy_mask = torch.zeros((img_height, img_width), dtype=torch.uint8)
                x1, y1, x2, y2 = bbox
                dummy_mask[int(y1):int(y2), int(x1):int(x2)] = 1
                masks.append(dummy_mask)
        
        # Konvertiere zu Tensoren mit Validierung
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack(masks) if masks else torch.zeros((0, img_height, img_width), dtype=torch.uint8)
        
        # Validiere Bounding Boxes
        if len(boxes) > 0:
            boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=img_width-1)   # x1
            boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=img_height-1)  # y1  
            boxes[:, 2] = torch.clamp(boxes[:, 2], min=1, max=img_width)     # x2
            boxes[:, 3] = torch.clamp(boxes[:, 3], min=1, max=img_height)    # y2
            
            # Stelle sicher, dass x2 > x1 und y2 > y1
            boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1)
            boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        
        return img, target

    def __len__(self):
        return len(self.metadata_cache)

# --- Modell-Setup ---
def get_model_instance_segmentation(num_classes):
    """Mask R-CNN Modell laden"""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Ersetze den Classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Ersetze den Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def collate_fn(batch):
    """Collate-Funktion für DataLoader"""
    return tuple(zip(*batch))

# --- HAUPTTRAINING ---
def main():
    print("🚀🚀🚀 MASK R-CNN RAM-TURBO TRAINING GESTARTET! 🚀🚀🚀")
    print(f"💾 System-RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"🔧 Konfiguration:")
    print(f"   - Batch Size: {BATCH_SIZE} (TURBO!)")
    print(f"   - Epochen: {NUM_EPOCHS}")
    print(f"   - Klassen: {NUM_CLASSES}")
    print(f"   - Device: {DEVICE}")
    print(f"   - RAM Cache: {RAM_CACHE_SIZE_GB}GB geplant")
    
    start_ram = get_ram_usage()
    print(f"🔄 Start-RAM: {start_ram:.1f}GB")
    
    # Dataset mit RAM-Cache laden
    dataset = RAMTurboMaskDataset(DATA_DIR, MASKS_DIR)
    
    if len(dataset) == 0:
        print("❌ Keine Trainingsdaten gefunden!")
        return
    
    # DataLoader mit höherem Batch Size
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # RAM-Cache braucht keine Workers
        pin_memory=False  # CPU Training
    )
    
    print(f"📊 RAM-TURBO Dataset: {len(dataset)} Bilder")
    print(f"📦 Batches pro Epoche: {len(data_loader)}")
    
    # Modell erstellen
    print(f"🤖 Lade Mask R-CNN Modell...")
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizer mit TURBO-Settings
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)  # Höhere LR dank Stabilität
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training Loop
    model.train()
    print(f"\n🎯 STARTE RAM-TURBO TRAINING...")
    training_start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        print(f"\n📈 Epoche {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            batch_start = time.time()
            
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # NaN-Check
            if not torch.isfinite(losses):
                print(f"⚠️ NaN/Inf Loss in Batch {batch_idx + 1}! Überspringe...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            batch_time = time.time() - batch_start
            
            # Progress-Ausgabe - häufiger wegen höherer Geschwindigkeit
            if (batch_idx + 1) % 5 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                ram_gb = get_ram_usage()
                print(f"   Batch {batch_idx + 1}/{len(data_loader)}: "
                      f"Loss = {losses.item():.4f}, "
                      f"Avg = {avg_loss:.4f}, "
                      f"Zeit = {batch_time:.2f}s, "
                      f"RAM = {ram_gb:.1f}GB")
        
        # Epoche abgeschlossen
        lr_scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(data_loader)
        
        print(f"✅ Epoche {epoch + 1} abgeschlossen:")
        print(f"   Durchschnittlicher Loss: {avg_epoch_loss:.4f}")
        print(f"   Zeit: {epoch_time:.2f} Sekunden")
        print(f"   Batches/Sekunde: {len(data_loader)/epoch_time:.2f}")
        
        # Modell speichern
        if (epoch + 1) % 2 == 0:
            model_path = OUTPUT_DIR / f"mask_rcnn_ram_turbo_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Modell gespeichert: {model_path}")
    
    # Finales Modell speichern
    final_model_path = OUTPUT_DIR / "mask_rcnn_ram_turbo_final.pth"
    torch.save(model.state_dict(), final_model_path)
    
    total_time = time.time() - training_start_time
    final_ram = get_ram_usage()
    
    print(f"\n" + "="*70)
    print(f"🎯🚀 RAM-TURBO TRAINING ERFOLGREICH ABGESCHLOSSEN! 🚀🎯")
    print(f"⏱️ Gesamtzeit: {total_time:.1f} Sekunden")
    print(f"💾 Finale RAM-Nutzung: {final_ram:.1f}GB")
    print(f"🏃 Durchschnittliche Epoche: {total_time/NUM_EPOCHS:.1f}s")
    print(f"💾 Finales Modell: {final_model_path}")
    print("="*70)

if __name__ == "__main__":
    main()