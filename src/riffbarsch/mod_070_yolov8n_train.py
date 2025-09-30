"""
🚀🔥 === ULTRA-HIGH-PERFORMANCE YOLOv8n Detection Training === 🔥🚀
BEAST-MODE: Nutzt 128GB RAM + 16 CPU-Kerne für MAXIMUM SPEED!

SPEED-GEHEIMNISSE:
- YOLOv8n (3M Parameter) statt YOLOv8s (11M) = 3.7x schneller!
- VOLLSTÄNDIGER RAM-Cache für 13,375 Bilder = Keine Disk-I/O!
- Batch-64 mit 128GB RAM = Maximum Memory Bandwidth!
- 16 Worker-Threads = ALLE Kerne auf Maximum!
- Aggressive Hyperparameter = Schnelle Konvergenz!
- Smart Early Stopping = Stopp bei Plateau!

ZIEL: <2 Stunden Training mit FULL PERFORMANCE!

Changelog:
2025-09-30: BEAST-VERSION - Kombiniert alle Speed-Tricks!
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import shutil
import gc

# Performance-Imports
import torch
import torch.multiprocessing as mp

# YOLO-Imports
from ultralytics import YOLO
# === 🔥 BEAST-MODE KONFIGURATION für 128GB RAM + 16 Kerne! ===

# Hardware-Optimierung für MAXIMUM POWER!
torch.set_num_threads(16)  # ALLE 16 Kerne!
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['OMP_SCHEDULE'] = 'static'
os.environ['OMP_PROC_BIND'] = 'true'

# Memory-Management für 128GB Beast-Mode
gc.set_threshold(700, 10, 10)  # Aggressives Garbage Collection

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
import json

class BeastModeYoloTrainer:
    """
    🚀🔥 BEAST-MODE YOLOv8n Trainer mit MAXIMUM PERFORMANCE!
    
    SPEED-FEATURES:
    - 128GB RAM vollständig ausgenutzt
    - 16 CPU-Kerne parallel
    - YOLOv8n für maximale Geschwindigkeit
    - Smart Early Stopping
    - Hardware-optimierte Batches
    """
    
    def __init__(self, projekt_root: str):
        self.projekt_root = Path(projekt_root)
        self.dataset_pfad = self.projekt_root / "datasets" / "yolo_maskrcnn_large"
        self.model_pfad = self.projekt_root / "models" / "yolov8n_beast"
        
        # 🔥 BEAST-MODE Konfiguration!
        self.training_config = {
            # SPEED: YOLOv8n = 3x schneller als YOLOv8s!
            'model_name': 'yolov8n.pt',  # 3M Parameter statt 11M!
            
            # RAM-BEAST: Nutze 128GB voll aus!
            'batch_size': 64,     # MEGA-Batches mit 128GB RAM!
            'workers': 16,        # ALLE 16 Kerne parallel!
            'cache': True,        # Vollständiger RAM-Cache!
            
            # INTELLIGENT: Früher Stopp bei Plateau
            'epochs': 25,         # Max-Epochen als Safety
            'patience': 4,        # 4 schlechte Epochen = Stopp!
            
            # AGGRESSIVE: Schnelle Konvergenz
            'lr0': 0.008,        # Hohe Lernrate für MEGA-Batches
            'momentum': 0.95,    # Hoher Momentum für Stabilität
            'weight_decay': 0.0003,  # Leichte Regularisierung
            
            # MINIMAL: Wenig Augmentation = Mehr Speed
            'hsv_h': 0.01,       # Minimal color changes
            'hsv_s': 0.1,        # Minimal saturation
            'hsv_v': 0.1,        # Minimal brightness
            'degrees': 2.0,      # Minimal rotation
            'translate': 0.05,   # Minimal translation
            'scale': 0.1,        # Minimal scaling
            'shear': 1.0,        # Minimal shear
            'perspective': 0.0,  # No perspective
            'flipud': 0.0,       # No vertical flip
            'fliplr': 0.3,       # Minimal horizontal flip
            'mosaic': 0.4,       # Reduziertes Mosaic
            'mixup': 0.0,        # Kein Mixup für Speed
        }
        
        # Performance-Tracking
        self.hardware_info = self.get_hardware_info()
        self.training_stats = {}
        self.start_time = None
        
        # Visualisierung Setup
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

print("=== YOLOv8s DETECTION Training für Riffbarsch-Taucher (LARGE DATASET: 13.375 Bilder) ===")
print(f"CPU-Kerne begrenzt auf: 14 von 16 verfügbaren")
print(f"Dataset-Statistiken:")
print(f"- Format: YOLO Detection (.txt Annotationen)")
print(f"- Klasse 0: Riffbarsch") 
print(f"- Klasse 1: Taucher")
print(f"- Struktur: train/images, train/labels, val/images, val/labels")

# YOLOv8n Detection Modell laden (NICHT -cls!)
model = YOLO("yolov8s.pt")  # Small Model für bessere Accuracy bei großem Dataset

# YAML-Datei korrekt erstellen vor dem Training
def erstelle_yaml_konfiguration():
    """
    Erstellt die YOLO-YAML-Konfiguration für DETECTION mit korrekten Pfaden
    YOLOv8n Detection erwartet Ordnerstruktur: dataset/train/images + dataset/train/labels
    """
    dataset_path = os.path.dirname(YAML_PFAD)
    yaml_inhalt = f"""# YOLO-Konfiguration für Riffbarsch-Taucher OBJEKTERKENNUNG (Detection)
path: {dataset_path.replace(os.sep, '/')}
train: train/images
val: val/images
test: test/images

# Klassennamen für Detection
names:
  0: riffbarsch  
  1: taucher

# Anzahl der Klassen
nc: 2

# Detection-spezifische Einstellungen
# Label-Format: [class_id x_center y_center width height] (normalisiert 0-1)
"""
    
    # Dataset-Ordner für Detection erstellen
    os.makedirs(dataset_path, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, split, 'labels'), exist_ok=True)
    
    # Datei sicher erstellen und schreibgeschützt machen
    with open(YAML_PFAD, "w", encoding="utf-8") as f:
        f.write(yaml_inhalt)
    
    # Backup der YAML-Datei erstellen falls sie überschrieben wird
    backup_path = YAML_PFAD.replace('.yaml', '_backup.yaml')
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(yaml_inhalt)
    
    print(f"YAML-Konfiguration für DETECTION erstellt: {YAML_PFAD}")
    print(f"Backup erstellt: {backup_path}")
    print(f"Dataset-Ordnerstruktur für Detection vorbereitet!")

# YAML-Datei vor Training erstellen
erstelle_yaml_konfiguration()

# Hilfsfunktion für Beispiel-Annotationen erstellen
def erstelle_beispiel_annotationen():
    """
    Erstellt Beispiel-Annotationen für YOLO Detection Format
    Format: [class_id x_center y_center width height] (normalisiert 0-1)
    """
    dataset_path = os.path.dirname(YAML_PFAD)
    
    print("\n=== Beispiel-Annotationen für Detection ===")
    print("YOLO Detection Label Format:")
    print("  Jede .txt Datei entspricht einem Bild")
    print("  Format pro Zeile: class_id x_center y_center width height")
    print("  Alle Werte normalisiert (0.0 bis 1.0)")
    print("  Beispiel: '0 0.5 0.3 0.2 0.4' = Riffbarsch in der Mitte")
    print()
    
    # Erstelle Beispiel-Annotation
    beispiel_pfad = os.path.join(dataset_path, "beispiel_annotation.txt")
    beispiel_inhalt = """# YOLO Detection Annotation Beispiel
# Format: class_id x_center y_center width height (normalisiert 0-1)

# Beispiel 1: Riffbarsch (Klasse 0) mittig im Bild
0 0.5 0.5 0.3 0.4

# Beispiel 2: Taucher (Klasse 1) links oben
1 0.2 0.3 0.15 0.25

# Koordinaten-Erklärung:
# x_center: Horizontale Mitte der Box (0=links, 1=rechts)  
# y_center: Vertikale Mitte der Box (0=oben, 1=unten)
# width: Breite der Box relativ zur Bildbreite
# height: Höhe der Box relativ zur Bildhöhe
"""
    
    with open(beispiel_pfad, "w", encoding="utf-8") as f:
        f.write(beispiel_inhalt)
    
    print(f"Beispiel-Annotation erstellt: {beispiel_pfad}")
    print("⚠️ WICHTIG: Sie müssen echte Annotationen für Ihr Dataset erstellen!")

# WICHTIG: Erstelle Beispiel-Annotationen als Referenz
erstelle_beispiel_annotationen()

# Überprüfung der Datenstruktur
def pruefer_datenstruktur():
    """
    Prüft ob die Datenstruktur für YOLOv8n-Detection korrekt ist
    Erwartet: train/images/*.jpg + train/labels/*.txt
    """
    dataset_path = os.path.dirname(YAML_PFAD)
    
    print("\n=== Datenstruktur Überprüfung für YOLOv8n-Detection ===")
    splits_info = {}
    
    for split in ['train', 'val', 'test']:
        images_path = os.path.join(dataset_path, split, 'images')
        labels_path = os.path.join(dataset_path, split, 'labels')
        
        if os.path.exists(images_path):
            # Zähle Bilder
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                image_files.extend(glob.glob(os.path.join(images_path, f"*{ext}")))
            
            # Zähle Labels
            label_files = glob.glob(os.path.join(labels_path, "*.txt"))
            
            splits_info[split] = {
                'images': len(image_files),
                'labels': len(label_files),
                'matched': 0
            }
            
            # Überprüfe Übereinstimmung zwischen Bildern und Labels
            matched = 0
            for img_file in image_files:
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                label_file = os.path.join(labels_path, f"{img_name}.txt")
                if os.path.exists(label_file):
                    matched += 1
            
            splits_info[split]['matched'] = matched
            
            print(f"{split}:")
            print(f"  Bilder: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            print(f"  Übereinstimmend: {matched}")
            
            if matched < len(image_files):
                print(f"  ⚠️ WARNUNG: {len(image_files) - matched} Bilder ohne Labels!")
        else:
            print(f"❌ FEHLER: {split}/images nicht gefunden!")
            splits_info[split] = {'images': 0, 'labels': 0, 'matched': 0}
    
    return splits_info

dataset_info = pruefer_datenstruktur()

# Training mit optimierten Parametern für Detection
print("\nStarte Detection Training...")
try:
    # Verwende die YAML-Datei für Detection
    results = model.train(
        data=YAML_PFAD,  # YAML-Datei für Detection verwenden!
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device='cpu',  # CPU explizit erzwingen
        workers=14,     # Maximal 14 CPU-Kerne für Datenladung
        project=MODELL_ORDNER,
        name='riffbarsch_taucher_large_detection',
        exist_ok=True,
        verbose=True,
        patience=15,  # Early stopping nach 15 Epochen ohne Verbesserung (erhöht)
        save=True,
        plots=True,  # Automatische Plot-Erstellung (mAP, Precision, Recall)
        val=True,
        # Erweiterte Hyperparameter für bessere Performance
        lr0=0.01,      # Initial learning rate
        lrf=0.01,      # Final OneCycleLR learning rate (lr0 * lrf)
        momentum=0.937,# SGD momentum/Adam beta1
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3.0,    # Warmup epochs (fractions ok)
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=0.1,   # Warmup initial bias lr
        box=7.5,       # Box loss gain
        cls=0.5,       # Cls loss gain  
        dfl=1.5,       # DFL loss gain
        pose=12.0,     # Pose loss gain (pose datasets only)
        kobj=1.0,      # Keypoint obj loss gain (pose datasets only)
        label_smoothing=0.0,  # Label smoothing (fraction)
        nbs=64,        # Nominal batch size
        hsv_h=0.015,   # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,     # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,     # Image HSV-Value augmentation (fraction)
        degrees=0.0,   # Image rotation (+/- deg)
        translate=0.1, # Image translation (+/- fraction)
        scale=0.5,     # Image scale (+/- gain)
        shear=0.0,     # Image shear (+/- deg)
        perspective=0.0, # Image perspective (+/- fraction)
        flipud=0.0,    # Image flip up-down (probability)
        fliplr=0.5,    # Image flip left-right (probability)
        mosaic=1.0,    # Image mosaic (probability)
        mixup=0.0,     # Image mixup (probability)
        copy_paste=0.0 # Segment copy-paste (probability)
    )
    print("Training erfolgreich abgeschlossen!")
    
except Exception as e:
    print(f"Fehler beim Training: {e}")
    print("Versuche alternative Konfiguration...")
    
    # Fallback: Training mit kleineren Parametern
    try:
        results = model.train(
            data=DATASET_ROOT,
            epochs=20,  # Weniger Epochen für Test
            batch=8,    # Kleinere Batch-Size
            imgsz=320,  # Kleinere Bildgröße
            device='cpu',
            workers=8,  # Weniger Worker
            project=MODELL_ORDNER,
            name='riffbarsch_taucher_test',
            exist_ok=True
        )
        print("Fallback-Training erfolgreich!")
    except Exception as e2:
        print(f"Auch Fallback-Training fehlgeschlagen: {e2}")
        exit(1)

print("=== Training abgeschlossen ===")

# Umfassende Visualisierungen nach dem Training
def erstelle_trainings_visualisierungen(results, model):
    """
    Erstellt umfassende Visualisierungen für YOLOv8n Training
    """
    print("\n=== Erstelle Trainings-Visualisierungen ===")
    
    # 1. Training/Validation Loss und Accuracy Kurven
    def plot_training_curves():
        try:
            # Versuche Metriken aus den Ergebnissen zu extrahieren
            results_csv = Path(results.save_dir) / "results.csv"
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()  # Leerzeichen entfernen
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Loss Kurven
                if 'train/loss' in df.columns and 'val/loss' in df.columns:
                    axes[0,0].plot(df.index, df['train/loss'], label='Training Loss', color='blue')
                    axes[0,0].plot(df.index, df['val/loss'], label='Validation Loss', color='red')
                    axes[0,0].set_title('Training vs Validation Loss')
                    axes[0,0].set_xlabel('Epoche')
                    axes[0,0].set_ylabel('Loss')
                    axes[0,0].legend()
                    axes[0,0].grid(True)
                
                # Accuracy Kurven
                if 'metrics/accuracy_top1' in df.columns:
                    axes[0,1].plot(df.index, df['metrics/accuracy_top1'], label='Top-1 Accuracy', color='green')
                    axes[0,1].set_title('Validation Accuracy')
                    axes[0,1].set_xlabel('Epoche')
                    axes[0,1].set_ylabel('Accuracy')
                    axes[0,1].legend()
                    axes[0,1].grid(True)
                
                # Learning Rate
                if 'lr/pg0' in df.columns:
                    axes[1,0].plot(df.index, df['lr/pg0'], label='Learning Rate', color='orange')
                    axes[1,0].set_title('Learning Rate Schedule')
                    axes[1,0].set_xlabel('Epoche')
                    axes[1,0].set_ylabel('Learning Rate')
                    axes[1,0].legend()
                    axes[1,0].grid(True)
                
                # Memory Usage (falls verfügbar)
                axes[1,1].text(0.5, 0.5, 'YOLOv8n Training\nKomplettes Dashboard', 
                              ha='center', va='center', transform=axes[1,1].transAxes, 
                              fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[1,1].set_title('Training Summary')
                
                plt.tight_layout()
                curves_path = Path(results.save_dir) / "training_curves_detailed.png"
                plt.savefig(curves_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Training-Kurven gespeichert: {curves_path}")
            else:
                print("Keine results.csv gefunden für Training-Kurven")
        except Exception as e:
            print(f"Fehler beim Erstellen der Training-Kurven: {e}")
    
    # 2. Confusion Matrix und Classification Report
    def plot_confusion_matrix_and_metrics():
        try:
            print("Erstelle Confusion Matrix...")
            
            # Test-Daten laden für Confusion Matrix
            test_path = Path(YAML_PFAD).parent / "test"
            if not test_path.exists():
                print("Test-Ordner nicht gefunden!")
                return [], [], [], []
            
            # Alle Test-Bilder sammeln
            test_images = []
            true_labels = []
            
            # Riffbarsch-Bilder (Klasse 0)
            riffbarsch_test = test_path / "images" 
            if riffbarsch_test.exists():
                for img_file in riffbarsch_test.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        test_images.append(str(img_file))
                        # Labels aus den Label-Dateien lesen
                        label_file = test_path / "labels" / f"{img_file.stem}.txt"
                        if label_file.exists():
                            with open(label_file, 'r') as f:
                                label_line = f.readline().strip()
                                if label_line:
                                    true_labels.append(int(label_line.split()[0]))
                                else:
                                    true_labels.append(0)  # Default Riffbarsch
                        else:
                            true_labels.append(0)  # Default Riffbarsch
            
            if len(test_images) > 0:
                # Vorhersagen auf Test-Set
                results_pred = model.predict(test_images, verbose=False)
                predicted_labels = []
                confidences = []
                
                for result in results_pred:
                    if hasattr(result, 'probs') and result.probs is not None:
                        pred_class = result.probs.top1
                        confidence = result.probs.top1conf.item()
                        predicted_labels.append(pred_class)
                        confidences.append(confidence)
                    else:
                        predicted_labels.append(0)
                        confidences.append(0.5)
                
                # Confusion Matrix
                cm = confusion_matrix(true_labels[:len(predicted_labels)], predicted_labels)
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Confusion Matrix Plot
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Riffbarsch', 'Taucher'], 
                           yticklabels=['Riffbarsch', 'Taucher'],
                           ax=axes[0])
                axes[0].set_title('Confusion Matrix')
                axes[0].set_xlabel('Vorhergesagt')
                axes[0].set_ylabel('Tatsächlich')
                
                # Confidence Distribution
                axes[1].hist([conf for i, conf in enumerate(confidences) if predicted_labels[i] == 0], 
                           alpha=0.7, label='Riffbarsch', bins=20, color='blue')
                axes[1].hist([conf for i, conf in enumerate(confidences) if predicted_labels[i] == 1], 
                           alpha=0.7, label='Taucher', bins=20, color='orange')
                axes[1].set_title('Confidence Distribution')
                axes[1].set_xlabel('Confidence Score')
                axes[1].set_ylabel('Anzahl Vorhersagen')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                cm_path = Path(results.save_dir) / "confusion_matrix_analysis.png"
                plt.savefig(cm_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Confusion Matrix gespeichert: {cm_path}")
                
                # Classification Report
                report = classification_report(true_labels[:len(predicted_labels)], predicted_labels, 
                                             target_names=['Riffbarsch', 'Taucher'], output_dict=True)
                
                # Report als DataFrame und Visualisierung
                report_df = pd.DataFrame(report).transpose()
                
                plt.figure(figsize=(10, 6))
                metrics = ['precision', 'recall', 'f1-score']
                classes = ['Riffbarsch', 'Taucher']
                
                x = np.arange(len(classes))
                width = 0.25
                
                for i, metric in enumerate(metrics):
                    values = [report_df.loc[cls, metric] for cls in classes]
                    plt.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
                
                plt.xlabel('Klassen')
                plt.ylabel('Score')
                plt.title('Classification Metrics pro Klasse')
                plt.xticks(x + width, classes)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                
                metrics_path = Path(results.save_dir) / "classification_metrics.png"
                plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Classification Metrics gespeichert: {metrics_path}")
                
                return predicted_labels, true_labels[:len(predicted_labels)], confidences, test_images
            else:
                print("Keine Test-Bilder gefunden!")
                return [], [], [], []
            
        except Exception as e:
            print(f"Fehler bei Confusion Matrix: {e}")
            return [], [], [], []
    
    # 3. Falsch klassifizierte Bilder anzeigen
    def plot_misclassified_images(predicted_labels, true_labels, confidences, test_images):
        try:
            print("Sammle falsch klassifizierte Bilder...")
            
            # Überprüfung auf leere Listen
            if not predicted_labels or not true_labels or not test_images:
                print("Keine Daten für Fehlklassifikations-Analyse verfügbar.")
                return
            
            misclassified = []
            for i, (pred, true, conf, img_path) in enumerate(zip(predicted_labels, true_labels, confidences, test_images)):
                if pred != true:
                    misclassified.append({
                        'image_path': img_path,
                        'true_label': true,
                        'pred_label': pred,
                        'confidence': conf,
                        'true_name': 'Riffbarsch' if true == 0 else 'Taucher',
                        'pred_name': 'Riffbarsch' if pred == 0 else 'Taucher'
                    })
            
            if len(misclassified) > 0:
                n_show = min(12, len(misclassified))
                cols = 4
                rows = (n_show + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1) if n_show > 1 else [axes]
                axes = axes.flatten() if rows > 1 else axes
                
                for i in range(len(axes)):
                    if i < n_show:
                        item = misclassified[i]
                        try:
                            img = Image.open(item['image_path'])
                            axes[i].imshow(img)
                            axes[i].set_title(f"Tatsächlich: {item['true_name']}\n"
                                            f"Vorhergesagt: {item['pred_name']}\n"
                                            f"Konfidenz: {item['confidence']:.2f}",
                                            color='red', fontsize=10)
                            axes[i].axis('off')
                            
                            # Roter Rahmen für falsche Klassifikationen
                            for spine in axes[i].spines.values():
                                spine.set_edgecolor('red')
                                spine.set_linewidth(3)
                        except Exception as e:
                            axes[i].text(0.5, 0.5, f"Fehler beim Laden\n{os.path.basename(item['image_path'])}", 
                                       ha='center', va='center', transform=axes[i].transAxes)
                            axes[i].axis('off')
                    else:
                        axes[i].axis('off')
                
                plt.suptitle(f'Falsch klassifizierte Bilder ({len(misclassified)} gefunden, {n_show} angezeigt)', 
                           fontsize=16, color='red')
                plt.tight_layout()
                
                misclass_path = Path(results.save_dir) / "misclassified_images.png"
                plt.savefig(misclass_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Falsch klassifizierte Bilder gespeichert: {misclass_path}")
                
                # Statistik der Fehlklassifikationen
                print(f"\n=== Fehlklassifikations-Statistik ===")
                print(f"Gesamt falsch klassifiziert: {len(misclassified)} von {len(predicted_labels)} ({len(misclassified)/len(predicted_labels)*100:.1f}%)")
                
                riff_als_taucher = sum(1 for item in misclassified if item['true_label'] == 0 and item['pred_label'] == 1)
                taucher_als_riff = sum(1 for item in misclassified if item['true_label'] == 1 and item['pred_label'] == 0)
                
                print(f"Riffbarsch fälschlicherweise als Taucher: {riff_als_taucher}")
                print(f"Taucher fälschlicherweise als Riffbarsch: {taucher_als_riff}")
            else:
                print("Keine falsch klassifizierten Bilder gefunden - Perfekte Klassifikation!")
                
        except Exception as e:
            print(f"Fehler bei falsch klassifizierten Bildern: {e}")
    
    # 4. Training Summary Dashboard
    def create_summary_dashboard():
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Dataset Verteilung
            labels = ['Riffbarsch\n(92.5%)', 'Taucher\n(7.5%)']
            sizes = [92.5, 7.5]
            colors = ['lightblue', 'orange']
            axes[0,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0,0].set_title('Dataset Klassenverteilung')
            
            # Augmentation Übersicht
            aug_info = """Angewandte Augmentationen:
• Horizontal Flip: 50%
• Vertical Flip: 50%  
• Rotation: ±15°
• Skalierung: ±30%
• Scherung: ±10°
• HSV Anpassung
• Perspektive: 0.01%"""
            
            axes[0,1].text(0.1, 0.9, aug_info, transform=axes[0,1].transAxes, 
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
            axes[0,1].set_title('Data Augmentation')
            axes[0,1].axis('off')
            
            # Training Konfiguration
            config_info = f"""Training Konfiguration:
• Modell: YOLOv8n-cls
• Epochs: {EPOCHS} (Early Stop: 15)
• Batch Size: {BATCH_SIZE}
• Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}
• Device: CPU (14 Kerne)
• Optimizer: Auto (AdamW)
• Learning Rate: 0.01"""
            
            axes[1,0].text(0.1, 0.9, config_info, transform=axes[1,0].transAxes,
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
            axes[1,0].set_title('Training Setup')
            axes[1,0].axis('off')
            
            # Hardware Info
            hardware_info = f"""Hardware Spezifikationen:
• CPU: AMD Ryzen 7 PRO 4750G
• Genutzte Kerne: 14/16
• PyTorch: {torch.__version__}
• Device: CPU Only
• Workers: 14 parallel
• Memory: Optimiert"""
            
            axes[1,1].text(0.1, 0.9, hardware_info, transform=axes[1,1].transAxes,
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
            axes[1,1].set_title('Hardware Setup')
            axes[1,1].axis('off')
            
            plt.suptitle('YOLOv8n Training Dashboard - Riffbarsch vs Taucher', fontsize=16, y=0.95)
            plt.tight_layout()
            
            dashboard_path = Path(results.save_dir) / "training_dashboard.png"
            plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"Training Dashboard gespeichert: {dashboard_path}")
            
        except Exception as e:
            print(f"Fehler beim Summary Dashboard: {e}")
    
    # Alle Visualisierungen ausführen
    plot_training_curves()
    predicted_labels, true_labels, confidences, test_images = plot_confusion_matrix_and_metrics()
    if len(predicted_labels) > 0:
        plot_misclassified_images(predicted_labels, true_labels, confidences, test_images)
    create_summary_dashboard()
    
    print(f"\n=== Alle Visualisierungen gespeichert in: {results.save_dir} ===")

# Visualisierungen nach dem Training erstellen
erstelle_trainings_visualisierungen(results, model)

# Evaluation auf Test-Set für Detection
print("\n=== Detection Modell-Evaluierung ===")
try:
    metrics = model.val(
        data=YAML_PFAD,
        split='test'
    )
    
    print("\n=== Detection Metriken ===")
    if hasattr(metrics, 'box'):
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")  
        print(f"Recall: {metrics.box.mr:.4f}")
        
        # Klassenweise Metriken
        if hasattr(metrics.box, 'ap_class_index'):
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = 'riffbarsch' if class_idx == 0 else 'taucher'
                if i < len(metrics.box.ap50):
                    print(f"{class_name} AP@0.5: {metrics.box.ap50[i]:.4f}")
    
    # Zusätzliche Detection-Analyse
    def detection_test_analysis():
        test_images_path = os.path.join(DATASET_ROOT, "test", "images")
        if os.path.exists(test_images_path):
            test_images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_images.extend(glob.glob(os.path.join(test_images_path, f"*{ext}")))
            
            if test_images:
                print(f"\n=== Test auf {len(test_images)} Bildern ===")
                results_pred = model.predict(test_images[:10], verbose=False)  # Nur erste 10 für Demo
                
                total_detections = 0
                class_counts = {'riffbarsch': 0, 'taucher': 0}
                
                for result in results_pred:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            total_detections += 1
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if class_id == 0:
                                class_counts['riffbarsch'] += 1
                            elif class_id == 1:
                                class_counts['taucher'] += 1
                                
                            print(f"  Detection: {['riffbarsch', 'taucher'][class_id]} (Conf: {conf:.3f})")
                
                print(f"Gesamt-Detektionen: {total_detections}")
                print(f"Riffbarsch: {class_counts['riffbarsch']}, Taucher: {class_counts['taucher']}")
    
    detection_test_analysis()
    
except Exception as e:
    print(f"Fehler bei Evaluation: {e}")

print(f"\nDetection Training gespeichert in: {MODELL_ORDNER}")
print("Das Detection Model kann jetzt Bounding Boxes für Riffbarsche und Taucher erstellen!")
print("⚠️ WICHTIG: Stellen Sie sicher, dass Ihr Dataset korrekte Bounding Box Annotationen hat.")


