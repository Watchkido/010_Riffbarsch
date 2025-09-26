import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
from PIL import Image
import numpy as np
import json
from pathlib import Path
import time

# --- Parameter ---
DATA_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data")
MASKS_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data\masks")
OUTPUT_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\models\maskrcnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1  # Weiter reduziert für mehr Stabilität
NUM_EPOCHS = 5   # Reduziert für ersten Test
NUM_CLASSES = 3   # Hintergrund + riffbarsch + taucher (1, 2)

DEVICE = torch.device("cpu")  # Nur CPU verwenden
print(f"🖥️ Training auf: {DEVICE}")

# --- Dataset Klasse für individuelle Masken ---
class IndividualMaskDataset(Dataset):
    def __init__(self, root, masks_dir, transforms=None):
        self.root = Path(root)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        
        # Suche alle JSON-Metadaten-Dateien
        self.metadata_files = list(self.masks_dir.glob("train/*_metadata.json"))
        print(f"📊 Gefunden: {len(self.metadata_files)} Bilder mit Metadaten")

    def __getitem__(self, idx):
        metadata_file = self.metadata_files[idx]
        
        # Lade Metadaten
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Lade das Originalbild - suche in YOLO-Struktur
        img_name = metadata['image']
        # Suche das Bild in der YOLO-Struktur
        yolo_img_path = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_split\train\images") / img_name
        
        if not yolo_img_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {img_name}")
            
        img = Image.open(yolo_img_path).convert("RGB")
        img_width, img_height = img.size
        
        # Erstelle Listen für alle Instanzen
        boxes = []
        labels = []
        masks = []
        
        for instance in metadata['instances']:
            # Bounding Box
            bbox = instance['bbox']  # [x1, y1, x2, y2]
            boxes.append(bbox)
            
            # Label (class_id ist bereits +1 für Mask R-CNN Format)
            labels.append(instance['class_id'])
            
            # Lade individuelle Maske
            mask_filename = instance['mask_file']
            class_name = instance['class_name']
            mask_path = self.masks_dir / "train" / class_name / mask_filename
            
            if mask_path.exists():
                mask_img = Image.open(mask_path).convert("L")
                mask_array = np.array(mask_img) > 0  # Binäre Maske
                masks.append(mask_array)
            else:
                print(f"⚠️ Maske nicht gefunden: {mask_path}")
                # Erstelle Dummy-Maske aus Bounding Box
                dummy_mask = np.zeros((img_height, img_width), dtype=bool)
                x1, y1, x2, y2 = bbox
                dummy_mask[int(y1):int(y2), int(x1):int(x2)] = True
                masks.append(dummy_mask)
        
        # Konvertiere zu Tensoren (optimiert) mit Validierung
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Validiere Bounding Boxes gegen NaN/Inf
        if len(boxes) > 0:
            boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=img_width-1)   # x1
            boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=img_height-1)  # y1  
            boxes[:, 2] = torch.clamp(boxes[:, 2], min=1, max=img_width)     # x2
            boxes[:, 3] = torch.clamp(boxes[:, 3], min=1, max=img_height)    # y2
            
            # Stelle sicher, dass x2 > x1 und y2 > y1
            boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1)
            boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1)
        
        # Optimiert: erst zu numpy array, dann zu tensor
        masks_array = np.array(masks, dtype=np.uint8)
        masks = torch.as_tensor(masks_array, dtype=torch.uint8)
        
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
        
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            # Standard-Transform: PIL zu Tensor
            img = T.ToTensor()(img)
            
        return img, target

    def __len__(self):
        return len(self.metadata_files)

# --- Modell erstellen ---
def get_model_instance_segmentation(num_classes):
    # Lade vortrainiertes Mask R-CNN Modell
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Ersetze den Classifier im Box Predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Ersetze den Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# --- Collate-Funktion ---
def collate_fn(batch):
    return tuple(zip(*batch))

# --- Haupttraining ---
def main():
    print("🚀 MASK R-CNN TRAINING gestartet!")
    print(f"🔧 Konfiguration:")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Epochen: {NUM_EPOCHS}")
    print(f"   - Klassen: {NUM_CLASSES}")
    print(f"   - Device: {DEVICE}")
    
    # Dataset und DataLoader
    dataset = IndividualMaskDataset(DATA_DIR, MASKS_DIR)
    
    if len(dataset) == 0:
        print("❌ Keine Trainingsdaten gefunden!")
        return
    
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Für CPU-Training
    )
    
    print(f"📊 Trainingsdataset: {len(dataset)} Bilder")
    
    # Modell erstellen
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizer mit reduzierten Learning Rate für Stabilität
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)  # Reduzierte LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training Loop
    model.train()
    print(f"\n🎯 Starte Training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        print(f"\n📈 Epoche {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            batch_start = time.time()
            
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Prüfe auf NaN/Inf
            if not torch.isfinite(losses):
                print(f"⚠️ NaN/Inf Loss erkannt in Batch {batch_idx + 1}! Überspringe...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient Clipping gegen Exploding Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            batch_time = time.time() - batch_start
            
            # Progress-Ausgabe mit Loss-Validierung
            if (batch_idx + 1) % 10 == 0:  # Weniger häufig für bessere Performance
                if torch.isfinite(torch.tensor(epoch_loss)):
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"   Batch {batch_idx + 1}/{len(data_loader)}: "
                          f"Loss = {losses.item():.4f}, "
                          f"Avg = {avg_loss:.4f}, "
                          f"Zeit = {batch_time:.2f}s")
                else:
                    print(f"   Batch {batch_idx + 1}/{len(data_loader)}: NaN Loss erkannt!")
        
        # Epoche abgeschlossen
        lr_scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(data_loader)
        
        print(f"✅ Epoche {epoch + 1} abgeschlossen:")
        print(f"   Durchschnittlicher Loss: {avg_epoch_loss:.4f}")
        print(f"   Zeit: {epoch_time:.2f} Sekunden")
        
        # Modell speichern
        if (epoch + 1) % 2 == 0:
            model_path = OUTPUT_DIR / f"mask_rcnn_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Modell gespeichert: {model_path}")
    
    # Finales Modell speichern
    final_model_path = OUTPUT_DIR / "mask_rcnn_final.pth"
    torch.save(model.state_dict(), final_model_path)
    
    total_time = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"🎯 TRAINING ERFOLGREICH ABGESCHLOSSEN!")
    print(f"⏱️ Gesamtzeit: {total_time:.2f} Sekunden")
    print(f"💾 Finales Modell: {final_model_path}")
    print("="*60)

if __name__ == "__main__":
    main()