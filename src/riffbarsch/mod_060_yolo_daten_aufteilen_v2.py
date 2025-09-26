"""
Teilt die Bilder für YOLOv8n-Klassifikation auf:
- Erstellt Ordnerstruktur: dataset/split/klasse/bilder.jpg
- YOLOv8n-cls erwartet diese spezielle Struktur ohne Label-Dateien
- Aufteilung: 70% Train, 15% Val, 15% Test

Changelog:
- 2025-01-14: Erstellt für YOLOv8n-Klassifikation (ohne Labels)
- TODO: Klassen-Balance überprüfen und Augmentierung bei Imbalance

Humorvoller Kommentar: Wer seine Daten richtig sortiert, findet seine Fische auch im Dunkeln!
"""
import os
import shutil
import random
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict

# Konfiguration aus config.py importieren
from config import CONFIG

# Quellordner
RIFFBARSCH_ORDNER = [
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch",
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_extra", 
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\riffbarsch_i_net_data_augmentation"
]

TAUCHER_ORDNER = [
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher",
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\taucher_extra"
]

ZIEL_ROOT = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_classification"
SPLIT_RATIO = [0.7, 0.15, 0.15]  # Train, Val, Test
SPLIT_NAMES = ["train", "val", "test"]
BILDENDUNGEN = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Klassen-Mapping für YOLOv8n-Klassifikation
KLASSEN = {
    "riffbarsch": 0,
    "taucher": 1
}

def sammle_bilder_nach_klasse() -> Dict[str, List[str]]:
    """
    Sammelt alle Bilder nach Klassen sortiert
    
    :returns: Dictionary mit Klassennamen als Keys und Bildpfaden als Values
    :rtype: Dict[str, List[str]]
    """
    bilder_nach_klasse = {}
    
    # Riffbarsch-Bilder sammeln
    riffbarsch_bilder = []
    for ordner in RIFFBARSCH_ORDNER:
        if os.path.exists(ordner):
            for ext in BILDENDUNGEN:
                riffbarsch_bilder.extend(glob(os.path.join(ordner, "**", f"*{ext}"), recursive=True))
    
    random.shuffle(riffbarsch_bilder)
    bilder_nach_klasse["riffbarsch"] = riffbarsch_bilder
    
    # Taucher-Bilder sammeln
    taucher_bilder = []
    for ordner in TAUCHER_ORDNER:
        if os.path.exists(ordner):
            for ext in BILDENDUNGEN:
                taucher_bilder.extend(glob(os.path.join(ordner, "**", f"*{ext}"), recursive=True))
    
    random.shuffle(taucher_bilder)
    bilder_nach_klasse["taucher"] = taucher_bilder
    
    return bilder_nach_klasse

def erstelle_yolo_klassifikation_struktur(bilder_nach_klasse: Dict[str, List[str]]) -> None:
    """
    Erstellt YOLOv8n-Klassifikationsstruktur: dataset/split/klasse/
    
    :param bilder_nach_klasse: Dictionary mit Klassennamen und Bildpfaden
    :type bilder_nach_klasse: Dict[str, List[str]]
    """
    # Alten Zielordner komplett löschen und neu erstellen
    if os.path.exists(ZIEL_ROOT):
        shutil.rmtree(ZIEL_ROOT)
    
    # Für jede Klasse die Aufteilung durchführen
    splits_info = {}
    
    for klasse, bilder in bilder_nach_klasse.items():
        n_total = len(bilder)
        n_train = int(n_total * SPLIT_RATIO[0])
        n_val = int(n_total * SPLIT_RATIO[1])
        n_test = n_total - n_train - n_val
        
        splits = {
            "train": bilder[:n_train],
            "val": bilder[n_train:n_train+n_val],
            "test": bilder[n_train+n_val:]
        }
        
        splits_info[klasse] = {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": n_total
        }
        
        # Für jeden Split die Klassenordner erstellen und Bilder kopieren
        for split_name, split_bilder in splits.items():
            ziel_klassen_ordner = os.path.join(ZIEL_ROOT, split_name, klasse)
            os.makedirs(ziel_klassen_ordner, exist_ok=True)
            
            for bild_pfad in split_bilder:
                datei_name = os.path.basename(bild_pfad)
                ziel_pfad = os.path.join(ziel_klassen_ordner, datei_name)
                shutil.copy2(bild_pfad, ziel_pfad)
    
    return splits_info

def erstelle_yaml_konfiguration() -> str:
    """
    Erstellt YAML-Konfiguration für YOLOv8n-Klassifikation
    
    :returns: Pfad zur erstellten YAML-Datei
    :rtype: str
    """
    yaml_inhalt = f"""# YOLOv8n-Klassifikation für Riffbarsch-Taucher
# Erstellt: 2025-01-14 für YOLOv8n-cls Training
path: {ZIEL_ROOT.replace(os.sep, '/')}
train: train
val: val
test: test

# Klassen (werden automatisch aus Ordnerstruktur erkannt)
names:
  0: riffbarsch
  1: taucher

# Anzahl der Klassen
nc: 2
"""
    
    yaml_pfad = os.path.join(ZIEL_ROOT, "yolo_classification.yaml")
    with open(yaml_pfad, "w", encoding="utf-8") as f:
        f.write(yaml_inhalt)
    
    return yaml_pfad

def erstelle_datenset_uebersicht(bilder_nach_klasse: Dict[str, List[str]], splits_info: Dict) -> None:
    """
    Erstellt Übersichtsbild des Datasets mit Beispielbildern
    
    :param bilder_nach_klasse: Dictionary mit Klassennamen und Bildpfaden
    :type bilder_nach_klasse: Dict[str, List[str]]
    :param splits_info: Information über die Aufteilung
    :type splits_info: Dict
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("YOLOv8n-Klassifikation Dataset Übersicht", fontsize=16, fontweight="bold")
    
    # Beispielbilder für jede Klasse und jeden Split zeigen
    for i, (klasse, bilder) in enumerate(bilder_nach_klasse.items()):
        # Ein Beispielbild pro Split
        n_train = int(len(bilder) * SPLIT_RATIO[0])
        n_val = int(len(bilder) * SPLIT_RATIO[1])
        
        beispiele = {
            "Original": bilder[0] if bilder else None,
            "Train": bilder[0] if bilder else None,
            "Val": bilder[n_train] if len(bilder) > n_train else None,
            "Test": bilder[n_train + n_val] if len(bilder) > n_train + n_val else None
        }
        
        for j, (split_name, bild_pfad) in enumerate(beispiele.items()):
            ax = axes[i, j]
            
            if bild_pfad and os.path.exists(bild_pfad):
                try:
                    img = mpimg.imread(bild_pfad)
                    ax.imshow(img)
                    ax.set_title(f"{klasse.title()} - {split_name}", fontweight="bold")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Fehler beim Laden\n{str(e)}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{klasse.title()} - {split_name} (Fehler)")
            else:
                ax.text(0.5, 0.5, "Kein Bild\nverfügbar", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{klasse.title()} - {split_name} (Leer)")
            
            ax.axis('off')
    
    # Statistik-Text hinzufügen
    stats_text = "Dataset-Statistik:\n"
    for klasse, info in splits_info.items():
        stats_text += f"{klasse.title()}: {info['total']} Bilder\n"
        stats_text += f"  Train: {info['train']}, Val: {info['val']}, Test: {info['test']}\n"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Bild speichern
    uebersicht_pfad = os.path.join(ZIEL_ROOT, "dataset_uebersicht.png")
    plt.savefig(uebersicht_pfad, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dataset-Übersicht gespeichert: {uebersicht_pfad}")

def main() -> None:
    """
    Hauptfunktion: Erstellt YOLOv8n-Klassifikationsdataset
    """
    print("=== YOLOv8n-Klassifikation Dataset Erstellung ===")
    print("Struktur: dataset/split/klasse/ (ohne Label-Dateien)\n")
    
    # Schritt 1: Bilder sammeln
    print("1. Sammle Bilder nach Klassen...")
    bilder_nach_klasse = sammle_bilder_nach_klasse()
    
    for klasse, bilder in bilder_nach_klasse.items():
        print(f"   {klasse.title()}: {len(bilder)} Bilder gefunden")
    
    if not any(bilder_nach_klasse.values()):
        print("FEHLER: Keine Bilder gefunden! Überprüfe die Quellordner.")
        return
    
    # Schritt 2: YOLOv8n-Struktur erstellen
    print("\n2. Erstelle YOLOv8n-Klassifikationsstruktur...")
    splits_info = erstelle_yolo_klassifikation_struktur(bilder_nach_klasse)
    
    # Schritt 3: YAML-Konfiguration erstellen
    print("\n3. Erstelle YAML-Konfiguration...")
    yaml_pfad = erstelle_yaml_konfiguration()
    
    # Schritt 4: Übersicht erstellen
    print("\n4. Erstelle Dataset-Übersicht...")
    erstelle_datenset_uebersicht(bilder_nach_klasse, splits_info)
    
    # Abschlussbericht
    print(f"\n=== Dataset erfolgreich erstellt ===")
    print(f"Zielordner: {ZIEL_ROOT}")
    print(f"YAML-Datei: {yaml_pfad}")
    print(f"Struktur: dataset/{{train,val,test}}/{{riffbarsch,taucher}}/")
    
    # Finale Statistik
    total_bilder = sum(len(bilder) for bilder in bilder_nach_klasse.values())
    print(f"\nFinale Statistik:")
    print(f"Gesamt: {total_bilder} Bilder")
    
    for klasse, info in splits_info.items():
        anteil = (info['total'] / total_bilder) * 100
        print(f"{klasse.title()}: {info['total']} Bilder ({anteil:.1f}%)")
        print(f"  Train: {info['train']}, Val: {info['val']}, Test: {info['test']}")
    
    print(f"\nJetzt kann YOLOv8n-Klassifikation trainiert werden!")
    print(f"Befehl: python mod_020_yolov8n_train.py")

if __name__ == "__main__":
    main()