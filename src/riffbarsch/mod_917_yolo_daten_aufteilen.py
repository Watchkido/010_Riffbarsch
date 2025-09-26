"""
Teilt die Bilder für YOLOv8n Training auf:
- Riffbarsch-Bilder (Klasse 0) und Taucher-Bilder (Klasse 1)
- Aufteilung: 70% Train, 15% Val, 15% Test
- YOLO-Labeldateien werden mitkopiert

Humorvoller Kommentar: Wer seine Klassen richtig teilt, fischt im klaren Wasser!
"""
import os
import shutil
import random
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Quellordner
riffbarsch_ordner_1 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch"
riffbarsch_ordner_2 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_extra"
riffbarsch_ordner_3 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_i_net_data_augmentation"
taucher_ordner_1 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher"
taucher_ordner_2 = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher_extra"
ziel_root = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_split"

split_ratio = [0.7, 0.15, 0.15]  # Train, Val, Test
split_names = ["train", "val", "test"]
BILDENDUNGEN = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Riffbarsch-Bilder sammeln (Klasse 0)
riffbarsch_ordner = [riffbarsch_ordner_1, riffbarsch_ordner_2, riffbarsch_ordner_3]
riffbarsch_bilder = []
for ordner in riffbarsch_ordner:
    for ext in BILDENDUNGEN:
        riffbarsch_bilder += glob(os.path.join(ordner, "**", f"*{ext}"), recursive=True)
random.shuffle(riffbarsch_bilder)

# Taucher-Bilder sammeln (Klasse 1)
taucher_ordner = [taucher_ordner_1, taucher_ordner_2]
taucher_bilder = []
for ordner in taucher_ordner:
    for ext in BILDENDUNGEN:
        taucher_bilder += glob(os.path.join(ordner, "**", f"*{ext}"), recursive=True)
random.shuffle(taucher_bilder)

# Alle Bilder kombinieren
alle_bilder = [(img, 0) for img in riffbarsch_bilder] + [(img, 1) for img in taucher_bilder]
random.shuffle(alle_bilder)

n_total = len(alle_bilder)
n_train = int(n_total * split_ratio[0])
n_val   = int(n_total * split_ratio[1])
n_test  = n_total - n_train - n_val

splits = {
    "train": alle_bilder[:n_train],
    "val": alle_bilder[n_train:n_train+n_val],
    "test": alle_bilder[n_train+n_val:]
}

# Zielordner vorab leeren
for split in split_names:
    split_img_dir = os.path.join(ziel_root, split, "images")
    split_lbl_dir = os.path.join(ziel_root, split, "labels")
    
    # Alte Dateien löschen falls vorhanden
    for ordner in [split_img_dir, split_lbl_dir]:
        if os.path.exists(ordner):
            for datei in os.listdir(ordner):
                datei_pfad = os.path.join(ordner, datei)
                if os.path.isfile(datei_pfad):
                    os.remove(datei_pfad)
        os.makedirs(ordner, exist_ok=True)

for split, files_with_class in splits.items():
    split_img_dir = os.path.join(ziel_root, split, "images")
    split_lbl_dir = os.path.join(ziel_root, split, "labels")
    
    for img_path, klasse in files_with_class:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(os.path.dirname(img_path), base + ".txt")
        
        # Bild kopieren
        shutil.copy2(img_path, split_img_dir)
        
        # Label-Datei erstellen oder kopieren
        ziel_lbl_path = os.path.join(split_lbl_dir, base + ".txt")
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, split_lbl_dir)
        else:
            # YOLO-Label erstellen: Klasse x_center y_center width height (normiert)
            # Für Klassifikation verwenden wir das gesamte Bild
            with open(ziel_lbl_path, "w") as f:
                f.write(f"{klasse} 0.5 0.5 1.0 1.0\n")

print(f"YOLO-Datenset erstellt in: {ziel_root}")
print(f"Riffbarsch-Bilder: {len(riffbarsch_bilder)} (Klasse 0)")
print(f"Taucher-Bilder: {len(taucher_bilder)} (Klasse 1)")
print(f"Gesamt: {n_total} Bilder aufgeteilt in {n_train} Train, {n_val} Val, {n_test} Test")

# YAML-Konfigurationsdatei für YOLOv8n erstellen
yaml_inhalt = f"""# YOLO-Konfiguration für Riffbarsch-Taucher Klassifikation
path: {ziel_root}
train: train/images
val: val/images
test: test/images

# Klassennamen
names:
  0: riffbarsch
  1: taucher

# Anzahl der Klassen
nc: 2
"""

yaml_pfad = os.path.join(ziel_root, "yolo_riff.yaml")
with open(yaml_pfad, "w", encoding="utf-8") as f:
    f.write(yaml_inhalt)

print(f"YAML-Konfiguration erstellt: {yaml_pfad}")
print("YOLOv8n kann direkt mit diesem Dataset trainieren!")

# Übersichtsbild der Quellordner erstellen
def erstelle_uebersichtsbild():
    # Basis-Ordner (ohne Unterordner)
    basis_ordner = {
        "riffbarsch": riffbarsch_ordner_1,
        "taucher": taucher_ordner_1
    }
    
    # Ordner mit Unterordnern
    ordner_mit_unterordnern = {
        "riffbarsch_extra": riffbarsch_ordner_2,
        "riffbarsch_i_net_data_augmentation": riffbarsch_ordner_3,
        "taucher_extra": taucher_ordner_2
    }
    
    # Alle Ordner und Unterordner sammeln
    alle_ordner_beispiele = []
    
    # Basis-Ordner hinzufügen
    for ordner_name, ordner_pfad in basis_ordner.items():
        beispiel_bilder = []
        for ext in BILDENDUNGEN:
            beispiel_bilder.extend(glob(os.path.join(ordner_pfad, "**", f"*{ext}"), recursive=True))
        
        if beispiel_bilder:
            alle_ordner_beispiele.append((ordner_name, beispiel_bilder[0]))
    
    # Ordner mit Unterordnern durchgehen
    for haupt_ordner_name, haupt_ordner_pfad in ordner_mit_unterordnern.items():
        # Hauptordner selbst prüfen
        beispiel_bilder = []
        for ext in BILDENDUNGEN:
            beispiel_bilder.extend(glob(os.path.join(haupt_ordner_pfad, f"*{ext}"), recursive=False))
        
        if beispiel_bilder:
            alle_ordner_beispiele.append((haupt_ordner_name, beispiel_bilder[0]))
        
        # Unterordner durchgehen
        if os.path.exists(haupt_ordner_pfad):
            for unterordner in os.listdir(haupt_ordner_pfad):
                unterordner_pfad = os.path.join(haupt_ordner_pfad, unterordner)
                if os.path.isdir(unterordner_pfad):
                    unter_beispiele = []
                    for ext in BILDENDUNGEN:
                        unter_beispiele.extend(glob(os.path.join(unterordner_pfad, "**", f"*{ext}"), recursive=True))
                    
                    if unter_beispiele:
                        display_name = f"{haupt_ordner_name}/{unterordner}"
                        alle_ordner_beispiele.append((display_name, unter_beispiele[0]))
    
    if not alle_ordner_beispiele:
        print("Keine Beispielbilder gefunden!")
        return
    
    # Grid-Layout für die Bilder
    n_ordner = len(alle_ordner_beispiele)
    cols = 4
    rows = (n_ordner + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if n_ordner > 1 else [axes]
    elif n_ordner == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (ordner_name, bild_pfad) in enumerate(alle_ordner_beispiele):
        try:
            # Bild laden und anzeigen
            img = Image.open(bild_pfad)
            axes[i].imshow(img)
            axes[i].set_title(f"{ordner_name}\n({os.path.basename(bild_pfad)})", fontsize=10, pad=10)
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Fehler beim Laden\n{ordner_name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
            print(f"Fehler beim Laden von {bild_pfad}: {e}")
    
    # Leere Subplots verstecken
    for i in range(len(alle_ordner_beispiele), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Beispielbilder aus Quellordnern ({len(alle_ordner_beispiele)} Ordner/Unterordner)', fontsize=16, y=0.95)
    plt.tight_layout()
    
    # Bild speichern
    uebersicht_pfad = os.path.join(ziel_root, "quellordner_uebersicht.png")
    plt.savefig(uebersicht_pfad, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Übersichtsbild der Quellordner erstellt: {uebersicht_pfad}")
    print(f"Anzahl dargestellter Ordner/Unterordner: {len(alle_ordner_beispiele)}")

# Übersichtsbild erstellen
erstelle_uebersichtsbild()
