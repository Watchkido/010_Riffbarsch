#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Konvertiert das große maskrcnn_data Dataset (13.381 Bilder) zu YOLO Detection Format
für bessere Objekterkennung von Riffbarschen und Tauchern.

Das maskrcnn_data Dataset ist etwa 7x größer als das aktuelle yolo_split Dataset
und sollte deutlich bessere Detection-Ergebnisse liefern.
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

def konvertiere_maskrcnn_zu_yolo_detection():
    """
    Konvertiert das große maskrcnn_data Klassifikations-Dataset 
    zu YOLO Detection Format mit Bounding Box Labels.
    """
    
    print("=== MASKRCNN_DATA → YOLO DETECTION KONVERTER ===")
    
    # Pfade definieren
    SOURCE_ROOT = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\maskrcnn_data")
    TARGET_ROOT = Path(r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_maskrcnn_large")
    
    # Ziel-Dataset erstellen
    TARGET_ROOT.mkdir(exist_ok=True)
    
    # Splits definieren
    splits = ['train', 'val', 'test']
    classes = {'riffbarsch': 0, 'taucher': 1}
    
    total_converted = 0
    
    for split in splits:
        print(f"\n📁 Konvertiere {split.upper()} Split...")
        
        # Zielordner erstellen
        (TARGET_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
        (TARGET_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        split_converted = 0
        
        for class_name, class_id in classes.items():
            source_class_dir = SOURCE_ROOT / split / class_name
            
            if not source_class_dir.exists():
                print(f"⚠️ Warnung: {source_class_dir} nicht gefunden")
                continue
            
            image_files = list(source_class_dir.glob('*.[jJ][pP][gG]')) + \
                         list(source_class_dir.glob('*.[jJ][pP][eE][gG]')) + \
                         list(source_class_dir.glob('*.[pP][nN][gG]'))
            
            print(f"  🔄 {class_name}: {len(image_files)} Bilder")
            
            for img_file in tqdm(image_files, desc=f"  {class_name}"):
                # Bild kopieren
                target_img = TARGET_ROOT / split / 'images' / img_file.name
                shutil.copy2(img_file, target_img)
                
                # YOLO Detection Label erstellen
                # Format: class_id x_center y_center width height (alles normalisiert 0-1)
                # Für Klassifikations→Detection: Vollbild-Bounding Box (0 0.5 0.5 1.0 1.0)
                label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
                
                label_file = TARGET_ROOT / split / 'labels' / f"{img_file.stem}.txt"
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write(label_content)
                
                split_converted += 1
        
        print(f"  ✅ {split}: {split_converted} Bilder konvertiert")
        total_converted += split_converted
    
    # YAML Konfiguration erstellen
    yaml_config = {
        'path': str(TARGET_ROOT).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': 2,
        'names': ['riffbarsch', 'taucher']
    }
    
    yaml_file = TARGET_ROOT / 'yolo_maskrcnn_large.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write("# YOLO Detection Config für großes maskrcnn_data Dataset\n")
        f.write("# 13.381+ Bilder für verbesserte Objekterkennung\n\n")
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n🎉 KONVERTIERUNG ABGESCHLOSSEN!")
    print(f"📊 Gesamt konvertiert: {total_converted:,} Bilder")
    print(f"📁 Ziel-Dataset: {TARGET_ROOT}")
    print(f"📄 YAML-Config: {yaml_file}")
    
    # Statistiken erstellen
    print(f"\n📈 DATASET-STATISTIKEN:")
    for split in splits:
        img_count = len(list((TARGET_ROOT / split / 'images').glob('*.[jJ]*')))
        label_count = len(list((TARGET_ROOT / split / 'labels').glob('*.txt')))
        print(f"  {split.upper()}: {img_count:,} Bilder, {label_count:,} Labels")
    
    return TARGET_ROOT, yaml_file

def validiere_konvertierung(dataset_root):
    """Validiert die Konvertierung"""
    print(f"\n🔍 VALIDIERUNG:")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = dataset_root / split / 'images'
        label_dir = dataset_root / split / 'labels'
        
        images = list(img_dir.glob('*.[jJ]*'))
        labels = list(label_dir.glob('*.txt'))
        
        print(f"  {split}: {len(images)} Bilder ↔ {len(labels)} Labels")
        
        if len(images) != len(labels):
            print(f"  ⚠️ WARNUNG: Bilder/Labels Anzahl stimmt nicht überein!")
        else:
            print(f"  ✅ {split}: OK")
    
    # Sample Label prüfen
    sample_label = dataset_root / 'train' / 'labels' / (list(labels)[0].name if labels else 'dummy.txt')
    if sample_label.exists():
        with open(sample_label, 'r') as f:
            content = f.read().strip()
        print(f"\n📝 Beispiel-Label: {content}")

if __name__ == "__main__":
    try:
        dataset_root, yaml_file = konvertiere_maskrcnn_zu_yolo_detection()
        validiere_konvertierung(dataset_root)
        
        print(f"\n🚀 NÄCHSTE SCHRITTE:")
        print(f"1. Training-Skript auf neues Dataset umstellen")
        print(f"2. Epochen auf 50+ erhöhen") 
        print(f"3. YOLOv8s statt YOLOv8n verwenden")
        print(f"4. Performance-Vergleich durchführen")
        
    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        raise