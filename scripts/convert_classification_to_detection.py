# Konvertiert Classification Dataset zu Detection Dataset
import os
import shutil
import glob
from pathlib import Path
import random

def konvertiere_classification_zu_detection():
    """
    Konvertiert das bestehende Classification Dataset zu Detection Format
    - Kopiert Bilder von train/class_name/ nach train/images/
    - Erstellt Vollbild-Annotations (0 0.5 0.5 1.0 1.0) für jede Klasse
    """
    
    # Pfade definieren
    source_path = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_classification"
    target_path = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_detection"
    
    print("=== Konvertierung Classification → Detection ===")
    print(f"Quelle: {source_path}")
    print(f"Ziel: {target_path}")
    
    # Klassen-Mapping
    class_mapping = {
        'riffbarsch': 0,  # Klasse 0
        'taucher': 1      # Klasse 1
    }
    
    # Statistiken
    stats = {'train': {}, 'val': {}, 'test': {}}
    
    # Für jeden Split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_source = os.path.join(source_path, split)
        if not os.path.exists(split_source):
            print(f"⚠️ {split} Ordner nicht gefunden, überspringe...")
            continue
        
        # Target-Ordner erstellen
        images_target = os.path.join(target_path, split, 'images')
        labels_target = os.path.join(target_path, split, 'labels')
        os.makedirs(images_target, exist_ok=True)
        os.makedirs(labels_target, exist_ok=True)
        
        print(f"\n📂 Verarbeite {split}...")
        split_stats = {'riffbarsch': 0, 'taucher': 0}
        
        # Für jede Klasse
        for class_name in ['riffbarsch', 'taucher']:
            class_source = os.path.join(split_source, class_name)
            if not os.path.exists(class_source):
                print(f"   ⚠️ {class_name} Ordner nicht gefunden")
                continue
            
            class_id = class_mapping[class_name]
            
            # Alle Bilder dieser Klasse finden
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_source, ext)))
                image_files.extend(glob.glob(os.path.join(class_source, ext.upper())))
            
            print(f"   📸 {class_name}: {len(image_files)} Bilder gefunden")
            
            # Bilder kopieren und Annotations erstellen
            for i, image_path in enumerate(image_files):
                # Neuer Dateiname mit Klassen-Präfix für bessere Organisation
                file_ext = os.path.splitext(image_path)[1].lower()
                new_filename = f"{class_name}_{i:04d}{file_ext}"
                
                # Bild kopieren
                target_image = os.path.join(images_target, new_filename)
                shutil.copy2(image_path, target_image)
                
                # Entsprechende YOLO-Annotation erstellen
                # Vollbild-Annotation: Objekt nimmt das ganze Bild ein
                # Format: class_id x_center y_center width height (normalisiert)
                annotation_content = f"{class_id} 0.5 0.5 1.0 1.0\\n"  # Vollbild
                
                # Annotation-Datei erstellen
                label_filename = f"{os.path.splitext(new_filename)[0]}.txt"
                label_path = os.path.join(labels_target, label_filename)
                with open(label_path, 'w') as f:
                    f.write(annotation_content)
                
                split_stats[class_name] += 1
        
        stats[split] = split_stats
        print(f"   ✅ {split}: {sum(split_stats.values())} Bilder konvertiert")
    
    # Gesamtstatistik ausgeben
    print(f"\n=== Konvertierung Abgeschlossen ===")
    total_images = 0
    total_riffbarsch = 0
    total_taucher = 0
    
    for split, split_stats in stats.items():
        if split_stats:  # Nur wenn Daten vorhanden
            split_total = sum(split_stats.values())
            total_images += split_total
            total_riffbarsch += split_stats.get('riffbarsch', 0)
            total_taucher += split_stats.get('taucher', 0)
            print(f"{split:>6}: {split_total:>4} Bilder (Riffbarsch: {split_stats.get('riffbarsch', 0):>3}, Taucher: {split_stats.get('taucher', 0):>2})")
    
    print(f"{'─'*50}")
    print(f"Gesamt: {total_images:>4} Bilder (Riffbarsch: {total_riffbarsch:>3}, Taucher: {total_taucher:>2})")
    print(f"\n🎯 Detection Dataset bereit in: {target_path}")
    print(f"📝 Alle Bilder haben Vollbild-Annotationen (0.5 0.5 1.0 1.0)")
    
    # YAML-Pfad aktualisieren (falls nötig)
    yaml_path = os.path.join(target_path, "yolo_detection.yaml")
    print(f"🔧 YAML-Konfiguration: {yaml_path}")
    
    return total_images, total_riffbarsch, total_taucher

if __name__ == "__main__":
    # Sicherheitsabfrage
    response = input("🚨 WARNUNG: Dies überschreibt das Demo-Dataset! Fortfahren? (ja/nein): ")
    if response.lower() in ['ja', 'j', 'yes', 'y']:
        total_images, riffbarsch, taucher = konvertiere_classification_zu_detection()
        print(f"\\n🎉 Erfolgreich {total_images} Bilder konvertiert!")
        print(f"   Riffbarsch: {riffbarsch} Bilder")
        print(f"   Taucher: {taucher} Bilder")
        print(f"\\n▶️ Jetzt können Sie das Training starten:")
        print(f"   python mod_070_yolov8n_train.py")
    else:
        print("❌ Konvertierung abgebrochen.")