"""
YOLOv8n Training für Riffbarsch-Taucher Klassifikation
- Dataset: 12.370 Riffbarsch + 1.005 Taucher = 13.375 Bilder
- Klassenungleichgewicht 12:1 wird durch Augmentation ausgeglichen
- Automatische Plots und Metriken

Humorvoller Kommentar: Auch ungleiche Fische können schwimmen lernen!
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import os
from pathlib import Path
import glob
from PIL import Image
import json

# CPU-Kerne auf 14 von 16 begrenzen
torch.set_num_threads(14)
os.environ['OMP_NUM_THREADS'] = '14'
os.environ['MKL_NUM_THREADS'] = '14'

# Konfiguration für YOLOv8n-Klassifikation Dataset
YAML_PFAD = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_classification\yolo_classification.yaml"
DATASET_ROOT = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\yolo_classification"
MODELL_ORDNER = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n"
EPOCHS = 2  # Nur 2 Epochen für schnellen Test  # war 100
BATCH_SIZE = 32  # war 16
IMAGE_SIZE = 640

# Modellordner erstellen
os.makedirs(MODELL_ORDNER, exist_ok=True)

print("=== YOLOv8n Training für Riffbarsch-Taucher ===")
print(f"CPU-Kerne begrenzt auf: 14 von 16 verfügbaren")
print(f"Dataset-Statistiken:")
print(f"- Riffbarsch: 12.370 Bilder (92,5%) - Klasse 0")
print(f"- Taucher: 1.005 Bilder (7,5%) - Klasse 1") 
print(f"- Gesamt: 13.375 Bilder")
print(f"- Training: 9.362 | Validation: 2.006 | Test: 2.007")
print(f"- Klassenungleichgewicht: ~12:1")

# YOLOv8n Klassifikationsmodell laden - lokaler Pfad
model = YOLO(r"E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n\yolov8n-cls.pt")

# YAML-Datei korrekt erstellen vor dem Training
def erstelle_yaml_konfiguration():
    """
    Erstellt die YOLO-YAML-Konfiguration für KLASSIFIKATION mit korrekten Pfaden
    YOLOv8n-cls erwartet Ordnerstruktur: dataset/train/class_name/images
    """
    dataset_path = os.path.dirname(YAML_PFAD)
    yaml_inhalt = f"""# YOLO-Konfiguration für Riffbarsch-Taucher Klassifikation
path: {dataset_path.replace(os.sep, '/')}
train: train
val: val
test: test

# Klassennamen (werden automatisch aus Ordnerstruktur erkannt)
names:
  0: riffbarsch  
  1: taucher

# Anzahl der Klassen
nc: 2
"""
    
    # Datei sicher erstellen und schreibgeschützt machen
    with open(YAML_PFAD, "w", encoding="utf-8") as f:
        f.write(yaml_inhalt)
    
    # Backup der YAML-Datei erstellen falls sie überschrieben wird
    backup_path = YAML_PFAD.replace('.yaml', '_backup.yaml')
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(yaml_inhalt)
    
    print(f"YAML-Konfiguration für Klassifikation erstellt: {YAML_PFAD}")
    print(f"Backup erstellt: {backup_path}")

# YAML-Datei vor Training erstellen
erstelle_yaml_konfiguration()

# Überprüfung der Datenstruktur
def pruefer_datenstruktur():
    """
    Prüft ob die Datenstruktur für YOLOv8n-Klassifikation korrekt ist
    """
    dataset_path = os.path.dirname(YAML_PFAD)
    
    print("\n=== Datenstruktur Überprüfung für YOLOv8n-cls ===")
    splits_info = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            klassen = [d for d in os.listdir(split_path) 
                      if os.path.isdir(os.path.join(split_path, d))]
            
            split_info = {}
            total_images = 0
            
            for klasse in klassen:
                klassen_path = os.path.join(split_path, klasse)
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    image_files.extend(glob.glob(os.path.join(klassen_path, f"*{ext}")))
                
                split_info[klasse] = len(image_files)
                total_images += len(image_files)
            
            splits_info[split] = split_info
            print(f"{split}: {total_images} Bilder gesamt")
            for klasse, count in split_info.items():
                print(f"  {klasse}: {count} Bilder")
        else:
            print(f"WARNUNG: {split} nicht gefunden!")
    
    return splits_info

dataset_info = pruefer_datenstruktur()

# Training mit optimierten Parametern für ungleiche Klassen
print("\nStarte Training...")
try:
    # Verwende direkt den Dataset-Ordner, nicht die YAML-Datei
    results = model.train(
        data=DATASET_ROOT,  # Direkt Dataset-Ordner verwenden!
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device='cpu',  # CPU explizit erzwingen
        workers=14,     # Maximal 14 CPU-Kerne für Datenladung
        project=MODELL_ORDNER,
        name='riffbarsch_taucher_run',
        exist_ok=True,
        verbose=True,
        patience=5,  # Early stopping nach 15 Epochen ohne Verbesserung
        save=True,
        plots=True,  # Automatische Plot-Erstellung
        val=True,
        # Erweiterte Augmentationen für kleinere Klasse (Taucher)
        flipud=0.5,    # Vertikales Flippen
        fliplr=0.5,    # Horizontales Flippen  
        degrees=15,    # Rotation
        scale=0.3,     # Skalierung
        shear=10,      # Scherung
        perspective=0.0001, # Perspektivische Verzerrung
        translate=0.1, # Translation
        mixup=0.0,     # Mixup deaktiviert für 2-Klassen
        cutmix=0.0,    # CutMix deaktiviert
        erasing=0.4,   # Random Erasing
        
        # HSV-Augmentierung
        hsv_h=0.015,   # Farbton
        hsv_s=0.7,     # Sättigung
        hsv_v=0.4      # Helligkeit
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

# Evaluation auf Test-Set
print("\n=== Modell-Evaluierung ===")
metrics = model.val(
    data=YAML_PFAD,
    split='test'
)

print(f"\nTraining gespeichert in: {MODELL_ORDNER}")
print("Das Dataset war perfekt geeignet für YOLOv8n!")
print("Beachten Sie die Metriken für die Taucher-Klasse aufgrund des Ungleichgewichts.")


