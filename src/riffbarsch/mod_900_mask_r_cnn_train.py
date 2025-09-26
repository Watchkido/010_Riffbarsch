import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import os
from PIL import Image
import numpy as np
import json
from pathlib import Path

# --- Parameter ---
DATA_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data")
OUTPUT_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\models\maskrcnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
NUM_EPOCHS = 5
NUM_CLASSES = 3   # Hintergrund + riffbarsch + taucher

DEVICE = torch.device("cpu")  # Nur CPU verwenden

# --- Dataset Klasse ---
class SimpleMaskDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.imgs = list(sorted(self.root.glob("**/*.jpg")))  # Alle Bilder rekursiv laden

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        # Lade die zugehörige Maske
        mask_path = img_path.with_name(img_path.stem + "_mask.png")
        mask = Image.open(mask_path).convert("L")  # Maske als Graustufenbild
        mask = np.array(mask)

        # Extrahiere Bounding Boxes aus der Maske
        obj_ids = np.unique(mask)[1:]  # Hintergrund ignorieren (ID=0)
        masks = mask == obj_ids[:, None, None]
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# --- Transforms ---
transform = T.Compose([T.ToTensor()])

# --- DataLoader ---
train_dataset = SimpleMaskDataset(DATA_DIR / "train", transform)  # Alle Klassen in "train" laden
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# --- Modell ---
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)

model.to(DEVICE)

# --- Optimizer ---
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    print(f"\n=== Starte Epoche {epoch+1}/{NUM_EPOCHS} ===")
    for batch_idx, (imgs, targets) in enumerate(train_loader, start=1):
        # Entferne GPU-Übertragung
        imgs = list(img for img in imgs)
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        # Fortschritt innerhalb der Epoche anzeigen
        print(f"Epoche {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")

    print(f"=== Epoche {epoch+1} abgeschlossen, Durchschnittlicher Verlust: {epoch_loss/len(train_loader):.4f} ===")

# --- Modell speichern ---
torch.save(model.state_dict(), OUTPUT_DIR / "maskrcnn_riffbarsch_taucher.pth")
print("✅ Training abgeschlossen, Modell gespeichert.")
