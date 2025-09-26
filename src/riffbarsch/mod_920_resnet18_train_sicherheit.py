# pip install torch torchvision albumentations
import torch, torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import sys
from torchvision.models import ResNet18_Weights, ResNet50_Weights

# Transform (nur Standardisierung und Tensor)
train_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Korrekte Pfade für ImageFolder: Die Klassenordner müssen direkt unter train/ val/ test/ liegen
train_dir = r"E:/dev/projekt_python_venv/010_Riffbarsch/datasets/resnet/train"
val_dir   = r"E:/dev/projekt_python_venv/010_Riffbarsch/datasets/resnet/val"
test_dir  = r"E:/dev/projekt_python_venv/010_Riffbarsch/datasets/resnet/test"

# Die Struktur muss so aussehen:
# train/
#   riffbarsch/
#   hard_negatives/
# val/
#   riffbarsch/
#   hard_negatives/
# test/
#   riffbarsch/
#   hard_negatives/

# Verschiebe alle Bilder aus .../hard_negatives/images/ nach .../hard_negatives/
## Die Verschiebung aus yolo_split ist nicht mehr nötig, da die ResNet-Struktur bereits korrekt ist.
## Die Daten werden direkt aus datasets/resnet geladen.

train_ds = ImageFolder(train_dir, transform=train_transforms)
val_ds   = ImageFolder(val_dir, transform=val_transforms)
test_ds  = ImageFolder(test_dir, transform=val_transforms)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=14, pin_memory=True)
val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=14, pin_memory=True)
test_dl  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=14, pin_memory=True)

# Modellwahl: resnet18 oder resnet50
MODELL_TYP = "resnet18"  # Alternativ: "resnet50"


if MODELL_TYP == "resnet18":
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
elif MODELL_TYP == "resnet50":
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
else:
    raise ValueError("MODELL_TYP muss 'resnet18' oder 'resnet50' sein!")

# Bestimme die Anzahl der Klassen automatisch aus dem Trainingsdatensatz
num_classes = len(train_ds.classes)
model.fc = nn.Linear(num_ftrs, num_classes)  # num_classes z.B. 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
val_losses = []

MAX_TRAINING_MINUTES = 30  # Maximale Trainingszeit in Minuten
start_time = time.time()

MODELLNAME = "fisch"
VERSION = "v1"
LAUFZEIT_MIN = MAX_TRAINING_MINUTES
MODELLORDNER = r"E:/dev/projekt_python_venv/010_Riffbarsch/models/resnet"
os.makedirs(MODELLORDNER, exist_ok=True)


EARLY_STOPPING_PATIENCE = 5  # Anzahl der Epochen ohne Verbesserung bis Abbruch
best_val_loss = float('inf')
epochs_no_improve = 0

if __name__ == "__main__":
    try:
        # Trainingsloop (vereinfacht)
        for epoch in range(1, 31):
            elapsed = (time.time() - start_time) / 60
            print(f"[Fortschritt] Epoche {epoch}/30 | Vergangene Zeit: {elapsed:.1f} min | Zeitlimit: {MAX_TRAINING_MINUTES} min")
            if elapsed > MAX_TRAINING_MINUTES:
                print(f"Training abgebrochen nach {MAX_TRAINING_MINUTES} Minuten.")
                break
            model.train()
            running_loss = 0.0
            for batch_idx, (imgs, labels) in enumerate(train_dl):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward(); optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                if batch_idx % 10 == 0:
                    print(f"  [Train] Batch {batch_idx+1}/{len(train_dl)} | Aktueller Loss: {loss.item():.4f}")
            epoch_loss = running_loss / len(train_dl.dataset)
            train_losses.append(epoch_loss)

            # Validierung
            model.eval()
            val_running_loss = 0.0
            for val_idx, (imgs, labels) in enumerate(val_dl):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_running_loss += loss.item() * imgs.size(0)
                if val_idx % 10 == 0:
                    print(f"  [Val] Batch {val_idx+1}/{len(val_dl)} | Aktueller Loss: {loss.item():.4f}")
            val_loss = val_running_loss / len(val_dl.dataset)
            val_losses.append(val_loss)
            print(f"[Epoche {epoch}] Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}")

            # Early Stopping prüfen
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  [Early Stopping] Keine Verbesserung für {epochs_no_improve} Epoche(n).")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Training gestoppt durch Early Stopping nach {epoch} Epochen (Val Loss hat sich {EARLY_STOPPING_PATIENCE} Epochen nicht verbessert).")
                    break

        # Modell speichern mit Zeitstempel, Laufzeit, Version und Name
        now = datetime.now().strftime("%Y%m%d_%H%M")
        modell_dateiname = f"{MODELLNAME}_{VERSION}_lz{LAUFZEIT_MIN}_{now}_resnet.pt"
        modell_pfad = os.path.join(MODELLORDNER, modell_dateiname)
        torch.save(model.state_dict(), modell_pfad)
        print(f"Modell gespeichert als {modell_pfad}")

        # Trainingsverlauf als Grafik
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.title(f'Trainingsverlauf für: {modell_dateiname}')
        plt.legend()
        plt.grid(True)
        diagramm_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '.png'))
        plt.savefig(diagramm_pfad)
        plt.show()
        print(f"Trainingskurve gespeichert als {diagramm_pfad}")
    except KeyboardInterrupt:
        print("Manueller Abbruch durch Strg+C. Das Programm wird beendet.")
    finally:
        sys.exit(0)
