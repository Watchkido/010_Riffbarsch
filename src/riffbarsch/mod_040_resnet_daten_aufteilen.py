"""
Teilt die Bilder auf drei Klassen auf und verteilt sie auf Train/Val/Test:
- Riffbarsch: Positiv-Bilder (mit Fisch) (70% Train, 15% Val, 15% Test)
- Hard Negatives: Negativ-Bilder (ohne Fisch) (70% Train, 15% Val, 15% Test)
- Taucher: Taucher-Bilder (70% Train, 15% Val, 15% Test)
- Eindeutige Dateinamen durch Zeitstempel

Humorvoller Kommentar: Wer seine Klassen nicht trennt, fischt im Trüben!
"""
import os
import shutil
import random
from glob import glob
from datetime import datetime

KLASSEN = ["riffbarsch", "hard_negatives", "taucher"]
SPLITS = ["train", "val", "test"]
BILDENDUNGEN = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Quellordner
positiv_ordner_1 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch"
positiv_ordner_2 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_extra"
positiv_ordner_3 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_i_net_data_augmentation"
hard_negativ_ordner = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\hard_negativ"
taucher_ordner_1 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher"
taucher_ordner_2 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher_extra"
ziel_root = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\resnet"

split_ratio = [0.7, 0.15, 0.15]  # Train, Val, Test

# Positiv-Bilder sammeln
positiv_ordner = [positiv_ordner_1, positiv_ordner_2, positiv_ordner_3]
positiv_bilder = []
for ordner in positiv_ordner:
    for ext in BILDENDUNGEN:
        positiv_bilder += glob(os.path.join(ordner, "**", f"*{ext}"), recursive=True)
random.shuffle(positiv_bilder)

n_total = len(positiv_bilder)
n_train = int(n_total * split_ratio[0])
n_val   = int(n_total * split_ratio[1])
n_test  = n_total - n_train - n_val

splits = {
    "train": positiv_bilder[:n_train],
    "val": positiv_bilder[n_train:n_train+n_val],
    "test": positiv_bilder[n_train+n_val:]
}




# Zielordner vorab leeren
for split in SPLITS:
    for klasse in KLASSEN:
        klasse_dir = os.path.join(ziel_root, split, klasse)
        if os.path.exists(klasse_dir):
            for f in os.listdir(klasse_dir):
                pfad = os.path.join(klasse_dir, f)
                if os.path.isfile(pfad):
                    os.remove(pfad)
                elif os.path.isdir(pfad):
                    shutil.rmtree(pfad)
        else:
            os.makedirs(klasse_dir, exist_ok=True)

# ResNet-kompatible Struktur: <split>/<klasse>/<bilddateien>
for split, files in splits.items():
    riffbarsch_dir = os.path.join(ziel_root, split, "riffbarsch")
    for img_path in files:
        shutil.copy2(img_path, riffbarsch_dir)

# Hard Negatives: jetzt auch auf train, val, test verteilen, mit Datum und Zeit im Dateinamen
hard_neg_bilder = []
for ext in BILDENDUNGEN:
    hard_neg_bilder += glob(os.path.join(hard_negativ_ordner, "**", f"*{ext}"), recursive=True)
random.shuffle(hard_neg_bilder)

n_total_hard = len(hard_neg_bilder)
n_train_hard = int(n_total_hard * split_ratio[0])
n_val_hard   = int(n_total_hard * split_ratio[1])
n_test_hard  = n_total_hard - n_train_hard - n_val_hard

hard_splits = {
    "train": hard_neg_bilder[:n_train_hard],
    "val": hard_neg_bilder[n_train_hard:n_train_hard+n_val_hard],
    "test": hard_neg_bilder[n_train_hard+n_val_hard:]
}

for split, files in hard_splits.items():
    hard_neg_dir = os.path.join(ziel_root, split, "hard_negatives")
    os.makedirs(hard_neg_dir, exist_ok=True)
    for img_path in files:
        # Erstelle einen eindeutigen Dateinamen mit Datum und Zeit
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(img_path)
        new_filename = f"{timestamp}_{original_name}"
        shutil.copy2(img_path, os.path.join(hard_neg_dir, new_filename))

# Taucher-Bilder: sammeln und auf train, val, test verteilen
taucher_ordner = [taucher_ordner_1, taucher_ordner_2]
taucher_bilder = []
for ordner in taucher_ordner:
    for ext in BILDENDUNGEN:
        taucher_bilder += glob(os.path.join(ordner, "**", f"*{ext}"), recursive=True)
random.shuffle(taucher_bilder)

n_total_taucher = len(taucher_bilder)
n_train_taucher = int(n_total_taucher * split_ratio[0])
n_val_taucher   = int(n_total_taucher * split_ratio[1])
n_test_taucher  = n_total_taucher - n_train_taucher - n_val_taucher

taucher_splits = {
    "train": taucher_bilder[:n_train_taucher],
    "val": taucher_bilder[n_train_taucher:n_train_taucher+n_val_taucher],
    "test": taucher_bilder[n_train_taucher+n_val_taucher:]
}

for split, files in taucher_splits.items():
    taucher_dir = os.path.join(ziel_root, split, "taucher")
    os.makedirs(taucher_dir, exist_ok=True)
    for img_path in files:
        # Erstelle einen eindeutigen Dateinamen mit Datum und Zeit
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(img_path)
        new_filename = f"{timestamp}_{original_name}"
        shutil.copy2(img_path, os.path.join(taucher_dir, new_filename))

print(f"ResNet-Ordnerstruktur und Bildverschiebung abgeschlossen. IT-Witz: Warum können Fische so gut debuggen? Sie finden jeden Bug im Netz!")