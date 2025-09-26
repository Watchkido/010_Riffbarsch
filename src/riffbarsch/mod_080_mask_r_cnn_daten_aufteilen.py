import os
import shutil
import random
from pathlib import Path
from datetime import datetime

# Deine Quellordner
RIFFBARSCH_ORDNER = [
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch",
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_extra",
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_i_net_data_augmentation"
]

TAUCHER_ORDNER = [
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher",
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher_extra"
]

# Zielordner
OUTPUT_DIR = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data")

# Lösche den gesamten Zielordner falls er existiert, um Duplikate zu vermeiden
if OUTPUT_DIR.exists():
    print(f"🗑️  Lösche vorherigen Ordner: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
    print("✅ Vorherige Daten gelöscht!")

# Erstelle die Ordnerstruktur neu
for split in ["train", "val", "test"]:
    for cls in ["riffbarsch", "taucher"]:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def collect_images(source_dirs, prefix):
    all_images = []
    for src in source_dirs:
        for root, _, files in os.walk(src):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_images.append(os.path.join(root, f))
    return all_images

# Bilder einsammeln
riffbarsch_imgs = collect_images(RIFFBARSCH_ORDNER, "riffbarsch")
taucher_imgs = collect_images(TAUCHER_ORDNER, "taucher")

print(f"Riffbarsch-Bilder: {len(riffbarsch_imgs)}")
print(f"Taucher-Bilder: {len(taucher_imgs)}")

def split_and_copy(images, cls_name):
    random.shuffle(images)
    total = len(images)
    train_split = int(0.7 * total)
    val_split = int(0.15 * total)

    splits = {
        "train": images[:train_split],
        "val": images[train_split:train_split+val_split],
        "test": images[train_split+val_split:]
    }

    # Aktuelles Datum und Zeit im Format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    counter = 1
    for split, imgs in splits.items():
        for img in imgs:
            ext = os.path.splitext(img)[1].lower()
            new_name = f"{cls_name}_{timestamp}_{counter}{ext}"
            shutil.copy(img, OUTPUT_DIR / split / cls_name / new_name)
            counter += 1

# Verteilen
split_and_copy(riffbarsch_imgs, "riffbarsch")
split_and_copy(taucher_imgs, "taucher")

print("✅ Dataset erfolgreich vorbereitet!")
