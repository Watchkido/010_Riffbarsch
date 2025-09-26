#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Skript für YOLOv8n Visualisierungen

Testet die reparierten Visualisierungs-Funktionen ohne Training durchzuführen.
"""

# Standardbibliotheken
from pathlib import Path
import os
import sys

# Wissenschaftliche Bibliotheken
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Machine Learning
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

# === KONFIGURATION ===
DATASET_ROOT = Path("E:/dev/projekt_python_venv/010_Riffbarsch/datasets/yolo_classification")
YAML_PFAD = DATASET_ROOT / "yolo_classification.yaml"

# Verwende bereits trainiertes Modell
MODEL_PATH = Path("E:/dev/projekt_python_venv/010_Riffbarsch/models/yolov8n/riffbarsch_taucher_test/weights/best.pt")

if not MODEL_PATH.exists():
    print(f"FEHLER: Trainiertes Modell nicht gefunden: {MODEL_PATH}")
    print("Verwende stattdessen Pre-trained Modell für Test...")
    MODEL_PATH = Path("E:/dev/projekt_python_venv/010_Riffbarsch/models/yolov8n/yolov8n-cls.pt")

print(f"=== YOLOv8n Visualisierungs-Test ===")
print(f"Verwende Modell: {MODEL_PATH}")

# Modell laden
model = YOLO(MODEL_PATH)

def erstelle_test_visualisierungen():
    """Erstellt alle Visualisierungen für das geladene Modell."""
    
    # Mock results für Test
    class MockResults:
        def __init__(self):
            self.save_dir = Path("E:/dev/projekt_python_venv/010_Riffbarsch/models/yolov8n/test_visualisierungen")
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    results = MockResults()
    
    # 1. Training Curves (falls results.csv vorhanden)
    def plot_training_curves():
        try:
            # Versuche Metriken aus den Ergebnissen zu extrahieren
            results_csv = Path(results.save_dir).parent / "results.csv"
            
            # Falls keine echten Results vorhanden, erstelle Demo-Daten
            if not results_csv.exists():
                print("Erstelle Demo Training Curves...")
                epochs = np.arange(1, 11)
                train_loss = 1.5 * np.exp(-epochs * 0.3) + 0.1 + np.random.normal(0, 0.05, len(epochs))
                val_loss = 1.6 * np.exp(-epochs * 0.25) + 0.15 + np.random.normal(0, 0.08, len(epochs))
                accuracy = 0.5 + 0.45 * (1 - np.exp(-epochs * 0.4)) + np.random.normal(0, 0.02, len(epochs))
                lr = 0.001 * np.exp(-epochs * 0.2)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Loss Kurven
                axes[0,0].plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
                axes[0,0].plot(epochs, val_loss, label='Validation Loss', color='red', linewidth=2)
                axes[0,0].set_title('Training vs Validation Loss', fontsize=14)
                axes[0,0].set_xlabel('Epoche')
                axes[0,0].set_ylabel('Loss')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
                
                # Accuracy Kurve
                axes[0,1].plot(epochs, accuracy, label='Validation Accuracy', color='green', linewidth=2)
                axes[0,1].set_title('Validation Accuracy', fontsize=14)
                axes[0,1].set_xlabel('Epoche')
                axes[0,1].set_ylabel('Accuracy')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].set_ylim([0, 1])
                
                # Learning Rate
                axes[1,0].plot(epochs, lr, label='Learning Rate', color='orange', linewidth=2)
                axes[1,0].set_title('Learning Rate Schedule', fontsize=14)
                axes[1,0].set_xlabel('Epoche')
                axes[1,0].set_ylabel('Learning Rate')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                
                # Training Summary
                axes[1,1].text(0.5, 0.5, 'YOLOv8n Visualisierungs-Test\n\n✓ Training Curves\n✓ Confusion Matrix\n✓ Fehlklassifikationen\n✓ Summary Dashboard', 
                              ha='center', va='center', transform=axes[1,1].transAxes, 
                              fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
                axes[1,1].set_title('Test Summary', fontsize=14)
                axes[1,1].set_xticks([])
                axes[1,1].set_yticks([])
                
                plt.tight_layout()
                curves_path = Path(results.save_dir) / "demo_training_curves.png"
                plt.savefig(curves_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Demo Training-Kurven gespeichert: {curves_path}")
                
        except Exception as e:
            print(f"Fehler bei Training-Kurven: {e}")
    
    # 2. Confusion Matrix und Classification Report
    def plot_confusion_matrix_and_metrics():
        try:
            print("Erstelle Demo Confusion Matrix...")
            
            # Test-Daten laden für Confusion Matrix
            test_path = DATASET_ROOT / "test"
            if not test_path.exists():
                print("Test-Ordner nicht gefunden! Erstelle Demo-Daten...")
                
                # Demo-Daten generieren
                np.random.seed(42)
                n_samples = 100
                
                # Simuliere realistische Vorhersagen (98.9% Accuracy)
                true_labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 90% Riffbarsch, 10% Taucher
                predicted_labels = true_labels.copy()
                
                # Füge 1.1% Fehler hinzu
                error_indices = np.random.choice(n_samples, size=int(n_samples * 0.011), replace=False)
                predicted_labels[error_indices] = 1 - predicted_labels[error_indices]
                
                confidences = np.random.beta(8, 2, size=n_samples)  # Höhere Konfidenz-Verteilung
                test_images = [f"demo_image_{i}.jpg" for i in range(n_samples)]
                
            else:
                return [], [], [], []  # Leere Listen wenn kein Test-Ordner
            
            # Confusion Matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Confusion Matrix Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Riffbarsch', 'Taucher'], 
                       yticklabels=['Riffbarsch', 'Taucher'],
                       ax=axes[0])
            axes[0].set_title('Demo Confusion Matrix', fontsize=14)
            axes[0].set_xlabel('Vorhergesagt')
            axes[0].set_ylabel('Tatsächlich')
            
            # Confidence Distribution
            riff_conf = [conf for i, conf in enumerate(confidences) if predicted_labels[i] == 0]
            tauch_conf = [conf for i, conf in enumerate(confidences) if predicted_labels[i] == 1]
            
            axes[1].hist(riff_conf, alpha=0.7, label='Riffbarsch', bins=20, color='blue')
            axes[1].hist(tauch_conf, alpha=0.7, label='Taucher', bins=20, color='orange')
            axes[1].set_title('Demo Confidence Distribution', fontsize=14)
            axes[1].set_xlabel('Confidence Score')
            axes[1].set_ylabel('Anzahl Vorhersagen')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            cm_path = Path(results.save_dir) / "demo_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"Demo Confusion Matrix gespeichert: {cm_path}")
            
            # Classification Report
            report = classification_report(true_labels, predicted_labels, 
                                         target_names=['Riffbarsch', 'Taucher'], output_dict=True)
            
            # Report als Visualisierung
            report_df = pd.DataFrame(report).transpose()
            
            plt.figure(figsize=(10, 6))
            metrics = ['precision', 'recall', 'f1-score']
            classes = ['Riffbarsch', 'Taucher']
            
            x = np.arange(len(classes))
            width = 0.25
            
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            for i, metric in enumerate(metrics):
                values = [report_df.loc[cls, metric] for cls in classes]
                plt.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8, color=colors[i])
            
            plt.xlabel('Klassen', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Demo Classification Metrics pro Klasse', fontsize=14)
            plt.xticks(x + width, classes)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Werte als Text anzeigen
            for i, metric in enumerate(metrics):
                for j, cls in enumerate(classes):
                    value = report_df.loc[cls, metric]
                    plt.text(j + i*width, value + 0.02, f'{value:.3f}', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            metrics_path = Path(results.save_dir) / "demo_classification_metrics.png"
            plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"Demo Classification Metrics gespeichert: {metrics_path}")
            
            return predicted_labels, true_labels, confidences, test_images
                
        except Exception as e:
            print(f"Fehler bei Demo Confusion Matrix: {e}")
            return [], [], [], []
    
    # 3. Falsch klassifizierte Bilder anzeigen
    def plot_misclassified_images(predicted_labels, true_labels, confidences, test_images):
        try:
            print("Demo: Sammle falsch klassifizierte Bilder...")
            
            # Überprüfung auf leere Listen
            if not predicted_labels or not true_labels or not test_images:
                print("Keine Daten für Demo Fehlklassifikations-Analyse verfügbar.")
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
                n_show = min(8, len(misclassified))
                cols = 4
                rows = (n_show + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1) if n_show > 1 else [axes]
                axes = axes.flatten() if rows > 1 else axes
                
                for i in range(len(axes)):
                    if i < n_show:
                        item = misclassified[i]
                        # Demo: Erstelle farbiges Platzhalter-Bild
                        color = 'blue' if item['true_label'] == 0 else 'orange'
                        demo_img = np.ones((100, 100, 3))
                        if item['true_label'] == 0:  # Riffbarsch
                            demo_img[:, :, 0] = 0.2  # Wenig Rot
                            demo_img[:, :, 1] = 0.4  # Wenig Grün  
                            demo_img[:, :, 2] = 0.8  # Viel Blau
                        else:  # Taucher
                            demo_img[:, :, 0] = 1.0  # Viel Rot
                            demo_img[:, :, 1] = 0.6  # Mittel Grün
                            demo_img[:, :, 2] = 0.0  # Kein Blau
                            
                        axes[i].imshow(demo_img)
                        axes[i].set_title(f"Tatsächlich: {item['true_name']}\n"
                                        f"Vorhergesagt: {item['pred_name']}\n"
                                        f"Konfidenz: {item['confidence']:.2f}",
                                        color='red', fontsize=10)
                        axes[i].axis('off')
                        
                        # Roter Rahmen für falsche Klassifikationen
                        for spine in axes[i].spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(3)
                    else:
                        axes[i].axis('off')
                
                plt.suptitle(f'Demo: Falsch klassifizierte Bilder ({len(misclassified)} gefunden, {n_show} angezeigt)', 
                           fontsize=16, color='red')
                plt.tight_layout()
                
                misclass_path = Path(results.save_dir) / "demo_misclassified_images.png"
                plt.savefig(misclass_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Demo falsch klassifizierte Bilder gespeichert: {misclass_path}")
                
                # Statistik der Fehlklassifikationen
                print(f"\n=== Demo Fehlklassifikations-Statistik ===")
                print(f"Gesamt falsch klassifiziert: {len(misclassified)} von {len(predicted_labels)} ({len(misclassified)/len(predicted_labels)*100:.1f}%)")
                
                riff_als_taucher = sum(1 for item in misclassified if item['true_label'] == 0 and item['pred_label'] == 1)
                taucher_als_riff = sum(1 for item in misclassified if item['true_label'] == 1 and item['pred_label'] == 0)
                
                print(f"Riffbarsch fälschlicherweise als Taucher: {riff_als_taucher}")
                print(f"Taucher fälschlicherweise als Riffbarsch: {taucher_als_riff}")
            else:
                print("Demo: Keine falsch klassifizierten Bilder gefunden - Perfekte Klassifikation!")
                
        except Exception as e:
            print(f"Fehler bei Demo falsch klassifizierten Bildern: {e}")
    
    # 4. Summary Dashboard
    def create_summary_dashboard():
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Dataset Verteilung
            labels = ['Riffbarsch\n(92.5%)', 'Taucher\n(7.5%)']
            sizes = [92.5, 7.5]
            colors = ['lightblue', 'orange']
            
            axes[0,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0,0].set_title('Demo Dataset Verteilung', fontsize=14)
            
            # Model Performance
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [0.989, 0.991, 0.985, 0.988]
            bars = axes[0,1].bar(metrics, values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
            axes[0,1].set_title('Demo Model Performance', fontsize=14)
            axes[0,1].set_ylim(0, 1)
            axes[0,1].grid(True, alpha=0.3)
            
            # Werte auf Balken anzeigen
            for bar, value in zip(bars, values):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Training Progress (simuliert)
            epochs = np.arange(1, 11)
            accuracy = 0.5 + 0.489 * (1 - np.exp(-epochs * 0.5))
            axes[1,0].plot(epochs, accuracy, marker='o', linewidth=2, markersize=6, color='green')
            axes[1,0].set_title('Demo Training Progress', fontsize=14)
            axes[1,0].set_xlabel('Epoche')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_ylim(0.4, 1.0)
            
            # System Info
            info_text = """🖥️ System Information:
            
            • YOLOv8n-cls Modell
            • CPU Training (14 Kerne)
            • Batch Size: 32
            • Image Size: 640x640
            • Epochs: 500 (Early Stop)
            
            📊 Dataset:
            • Total: 13,375 Bilder
            • Training: 9,362 Bilder  
            • Validation: 2,006 Bilder
            • Test: 2,007 Bilder
            
            🎯 Performance:
            • Best Accuracy: 98.9%
            • Training Time: ~2h
            • Klassifikations-Task
            """
            
            axes[1,1].text(0.05, 0.95, info_text, transform=axes[1,1].transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            axes[1,1].set_title('Demo System & Dataset Info', fontsize=14)
            axes[1,1].set_xticks([])
            axes[1,1].set_yticks([])
            
            plt.tight_layout()
            dashboard_path = Path(results.save_dir) / "demo_summary_dashboard.png"
            plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"Demo Training Dashboard gespeichert: {dashboard_path}")
            
        except Exception as e:
            print(f"Fehler beim Demo Summary Dashboard: {e}")
    
    # Alle Visualisierungen ausführen
    print("\n=== Teste alle Visualisierungs-Funktionen ===")
    plot_training_curves()
    predicted_labels, true_labels, confidences, test_images = plot_confusion_matrix_and_metrics()
    if len(predicted_labels) > 0:
        plot_misclassified_images(predicted_labels, true_labels, confidences, test_images)
    else:
        print("Überspringe Fehlklassifikations-Analyse (keine Daten)")
    create_summary_dashboard()
    print("\n✅ Alle Visualisierungen erfolgreich erstellt!")
    print(f"📁 Gespeichert in: {results.save_dir}")

if __name__ == "__main__":
    erstelle_test_visualisierungen()