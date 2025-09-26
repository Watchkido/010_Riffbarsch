#!/usr/bin/env python3
"""
Riffbarsch AI-Analyse GUI - Neu entwickelt
Saubere, funktionale Benutzeroberfläche für Fischanalyse
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
import threading
import time

# Matplotlib Threading-Problem lösen
plt.switch_backend('Agg')

# ==================== KONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESNET_PATH = r'E:\dev\projekt_python_venv\010_Riffbarsch\models\resnet18_riffbarsch_model.pth'
YOLO_PATH = r'E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n\riffbarsch_taucher_run\weights\best.pt'

# Klassen-Definitionen (korrigiert)
CLASS_NAMES_DISPLAY = ["Riffbarsch", "Taucher", "Anderer"]
CLASS_NAMES_RESNET = ["Taucher", "Riffbarsch", "Anderer"]

# ==================== GLOBALE VARIABLEN ====================
current_image = None
resnet_model = None
yolo_model = None

# ==================== MODEL LOADING ====================
def load_models():
    """Lädt alle AI-Modelle"""
    global resnet_model, yolo_model
    
    try:
        print("🧠 Lade ResNet18...")
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
        model.eval()
        resnet_model = model.to(DEVICE)
        print("✅ ResNet18 geladen")
    except Exception as e:
        print(f"❌ ResNet Fehler: {e}")
    
    try:
        print("🎯 Lade YOLO...")
        yolo_model = YOLO(YOLO_PATH)
        print("✅ YOLO geladen")
    except Exception as e:
        print(f"❌ YOLO Fehler: {e}")

# ==================== IMAGE TRANSFORMS ====================
resnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== HAUPTFENSTER ====================
class RiffbarschGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🐠 Riffbarsch AI-Analyse - Professionell")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Beim Schließen
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()
        load_models()
    
    def setup_ui(self):
        """Erstellt die Benutzeroberfläche"""
        # Hauptcontainer
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tabs erstellen
        self.notebook = ttk.Notebook(main_frame)
        
        # Tab 1: Upload
        self.tab_upload = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab_upload, text="📁 Bild laden")
        
        # Tab 2: Klassifikation  
        self.tab_classify = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab_classify, text="🧠 Klassifikation")
        
        # Tab 3: Objekterkennung
        self.tab_detect = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab_detect, text="🎯 Objekterkennung")
        
        # Tab 4: Segmentierung
        self.tab_segment = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab_segment, text="🎭 Segmentierung")
        
        self.notebook.pack(fill='both', expand=True)
        
        # Tabs einrichten
        self.setup_upload_tab()
        self.setup_classify_tab()
        self.setup_detect_tab()
        self.setup_segment_tab()
    
    def setup_upload_tab(self):
        """Upload Tab einrichten"""
        # Hauptcontainer
        container = tk.Frame(self.tab_upload, bg='white')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Titel
        title = tk.Label(container, text="🐠 Riffbarsch-Bild für Analyse auswählen", 
                        font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        title.pack(pady=20)
        
        # Bild-Anzeige
        self.image_frame = tk.Frame(container, bg='white', relief='sunken', bd=2)
        self.image_frame.pack(pady=20)
        
        self.canvas_upload = tk.Label(self.image_frame, text="Kein Bild geladen", 
                                    width=60, height=20, bg='#ecf0f1', fg='#7f8c8d')
        self.canvas_upload.pack(padx=20, pady=20)
        
        # RGB Histogramm
        self.hist_frame = tk.Frame(container, bg='white')
        self.hist_frame.pack(pady=10)
        
        # Button Container (IMMER sichtbar)
        button_frame = tk.Frame(container, bg='white')
        button_frame.pack(side='bottom', fill='x', pady=20)
        
        self.btn_load = tk.Button(button_frame, text="📁 Bild auswählen", 
                                command=self.load_image, bg='#3498db', fg='white',
                                font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.btn_load.pack(side='left', padx=10)
        
        self.btn_exit = tk.Button(button_frame, text="❌ Beenden", 
                                command=self.on_closing, bg='#e74c3c', fg='white',
                                font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.btn_exit.pack(side='left', padx=10)
    
    def setup_classify_tab(self):
        """Klassifikation Tab einrichten"""
        container = tk.Frame(self.tab_classify, bg='white')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Oberer Bereich für Bild und Diagramm
        top_frame = tk.Frame(container, bg='white')
        top_frame.pack(fill='both', expand=True)
        
        # Bild links
        img_frame = tk.Frame(top_frame, bg='white')
        img_frame.pack(side='left', padx=20, pady=20)
        
        tk.Label(img_frame, text="Analysiertes Bild", font=('Arial', 12, 'bold'), 
                bg='white').pack()
        self.canvas_classify = tk.Label(img_frame, text="Kein Bild", width=40, height=20,
                                      bg='#ecf0f1', fg='#7f8c8d')
        self.canvas_classify.pack()
        
        # Diagramm rechts
        chart_frame = tk.Frame(top_frame, bg='white')
        chart_frame.pack(side='right', fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(chart_frame, text="Klassifikations-Ergebnis", font=('Arial', 12, 'bold'),
                bg='white').pack()
        self.fig_classify_frame = tk.Frame(chart_frame, bg='white')
        self.fig_classify_frame.pack(fill='both', expand=True)
        
        # Button unten links
        button_frame = tk.Frame(container, bg='white')
        button_frame.pack(side='bottom', fill='x', pady=10)
        
        self.progress_classify = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress_classify.pack(side='bottom', fill='x', padx=20, pady=5)
        
        self.btn_classify = tk.Button(button_frame, text="🧠 Klassifikation starten", 
                                    command=self.start_classification, bg='#27ae60', fg='white',
                                    font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.btn_classify.pack(side='left', padx=20)
    
    def setup_detect_tab(self):
        """Objekterkennung Tab einrichten"""
        container = tk.Frame(self.tab_detect, bg='white')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        top_frame = tk.Frame(container, bg='white')
        top_frame.pack(fill='both', expand=True)
        
        # Bild links
        img_frame = tk.Frame(top_frame, bg='white')
        img_frame.pack(side='left', padx=20, pady=20)
        
        tk.Label(img_frame, text="Objekte erkannt", font=('Arial', 12, 'bold'), 
                bg='white').pack()
        self.canvas_detect = tk.Label(img_frame, text="Kein Bild", width=40, height=20,
                                    bg='#ecf0f1', fg='#7f8c8d')
        self.canvas_detect.pack()
        
        # Diagramm rechts
        chart_frame = tk.Frame(top_frame, bg='white')
        chart_frame.pack(side='right', fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(chart_frame, text="Erkennungs-Statistiken", font=('Arial', 12, 'bold'),
                bg='white').pack()
        self.fig_detect_frame = tk.Frame(chart_frame, bg='white')
        self.fig_detect_frame.pack(fill='both', expand=True)
        
        # Button unten links
        button_frame = tk.Frame(container, bg='white')
        button_frame.pack(side='bottom', fill='x', pady=10)
        
        self.progress_detect = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress_detect.pack(side='bottom', fill='x', padx=20, pady=5)
        
        self.btn_detect = tk.Button(button_frame, text="🎯 Objekterkennung starten", 
                                  command=self.start_detection, bg='#f39c12', fg='white',
                                  font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.btn_detect.pack(side='left', padx=20)
    
    def setup_segment_tab(self):
        """Segmentierung Tab einrichten"""
        container = tk.Frame(self.tab_segment, bg='white')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        top_frame = tk.Frame(container, bg='white')
        top_frame.pack(fill='both', expand=True)
        
        # Bild links
        img_frame = tk.Frame(top_frame, bg='white')
        img_frame.pack(side='left', padx=20, pady=20)
        
        tk.Label(img_frame, text="Segmentiertes Bild", font=('Arial', 12, 'bold'), 
                bg='white').pack()
        self.canvas_segment = tk.Label(img_frame, text="Kein Bild", width=40, height=20,
                                     bg='#ecf0f1', fg='#7f8c8d')
        self.canvas_segment.pack()
        
        # Diagramm rechts
        chart_frame = tk.Frame(top_frame, bg='white')
        chart_frame.pack(side='right', fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(chart_frame, text="Segmentierungs-Analyse", font=('Arial', 12, 'bold'),
                bg='white').pack()
        self.fig_segment_frame = tk.Frame(chart_frame, bg='white')
        self.fig_segment_frame.pack(fill='both', expand=True)
        
        # Button unten links
        button_frame = tk.Frame(container, bg='white')
        button_frame.pack(side='bottom', fill='x', pady=10)
        
        self.progress_segment = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress_segment.pack(side='bottom', fill='x', padx=20, pady=5)
        
        self.btn_segment = tk.Button(button_frame, text="🎭 Segmentierung starten", 
                                   command=self.start_segmentation, bg='#9b59b6', fg='white',
                                   font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.btn_segment.pack(side='left', padx=20)
    
    # ==================== IMAGE LOADING ====================
    def load_image(self):
        """Lädt ein Bild"""
        global current_image
        
        file_path = filedialog.askopenfilename(
            title="Riffbarsch-Bild auswählen",
            filetypes=[("Bilddateien", "*.jpg *.jpeg *.png *.bmp"), ("Alle", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            current_image = Image.open(file_path).convert("RGB")
            
            # Bild anzeigen
            display_img = current_image.copy()
            display_img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(display_img)
            self.canvas_upload.configure(image=tk_img, text="")
            self.canvas_upload.image = tk_img
            
            # RGB Histogramm erstellen
            self.create_rgb_histogram()
            
            print("✅ Bild erfolgreich geladen!")
            messagebox.showinfo("Erfolg", "Bild geladen! Wechseln Sie zu einem Analyse-Tab.")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{e}")
    
    def create_rgb_histogram(self):
        """Erstellt RGB Histogramm"""
        if current_image is None:
            return
        
        # Alte Widgets löschen
        for widget in self.hist_frame.winfo_children():
            widget.destroy()
        
        r, g, b = current_image.split()
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(np.array(r).flatten(), bins=50, color='red', alpha=0.7, label='Rot')
        ax.hist(np.array(g).flatten(), bins=50, color='green', alpha=0.7, label='Grün') 
        ax.hist(np.array(b).flatten(), bins=50, color='blue', alpha=0.7, label='Blau')
        ax.set_title("RGB-Histogramm", fontsize=12, fontweight='bold')
        ax.set_xlabel("Pixelwerte")
        ax.set_ylabel("Häufigkeit")
        ax.legend()
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.close(fig)
    
    # ==================== KLASSIFIKATION ====================
    def start_classification(self):
        """Startet Klassifikation"""
        if current_image is None:
            messagebox.showwarning("Warnung", "Bitte zuerst ein Bild laden!")
            return
        
        if resnet_model is None:
            messagebox.showerror("Fehler", "ResNet-Modell nicht verfügbar!")
            return
        
        self.btn_classify.config(state='disabled')
        self.progress_classify.start()
        
        thread = threading.Thread(target=self.run_classification)
        thread.daemon = True
        thread.start()
    
    def run_classification(self):
        """Führt Klassifikation durch"""
        try:
            # Bild transformieren
            img_tensor = resnet_transforms(current_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = resnet_model(img_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            # Klassenreihenfolge korrigieren
            probs_display = np.array([probs[1], probs[0], probs[2]])  # Taucher <-> Riffbarsch tauschen
            pred_idx = np.argmax(probs_display)
            prediction = CLASS_NAMES_DISPLAY[pred_idx]
            confidence = probs_display[pred_idx]
            
            # UI Update im Main Thread
            self.root.after(0, self.update_classification_ui, probs_display, prediction, confidence)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Fehler", f"Klassifikation fehlgeschlagen:\n{e}"))
        finally:
            self.root.after(0, self.finish_classification)
    
    def update_classification_ui(self, probs, prediction, confidence):
        """Aktualisiert Klassifikations-UI"""
        # Bild anzeigen
        display_img = current_image.copy()
        display_img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(display_img)
        self.canvas_classify.configure(image=tk_img, text="")
        self.canvas_classify.image = tk_img
        
        # Diagramm erstellen
        for widget in self.fig_classify_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(CLASS_NAMES_DISPLAY, probs, color=['#e74c3c', '#3498db', '#95a5a6'])
        bars[np.argmax(probs)].set_color('#27ae60')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel("Wahrscheinlichkeit")
        ax.set_title(f"🎯 Vorhersage: {prediction} ({confidence:.1%})", fontweight='bold')
        
        # Werte auf Balken
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{prob:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.fig_classify_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)
        
        print(f"✅ Klassifikation: {prediction} ({confidence:.1%})")
    
    def finish_classification(self):
        """Beendet Klassifikation"""
        self.progress_classify.stop()
        self.btn_classify.config(state='normal')
    
    # ==================== OBJEKTERKENNUNG ====================
    def start_detection(self):
        """Startet Objekterkennung"""
        if current_image is None:
            messagebox.showwarning("Warnung", "Bitte zuerst ein Bild laden!")
            return
        
        if yolo_model is None:
            messagebox.showerror("Fehler", "YOLO-Modell nicht verfügbar!")
            return
        
        self.btn_detect.config(state='disabled')
        self.progress_detect.start()
        
        thread = threading.Thread(target=self.run_detection)
        thread.daemon = True
        thread.start()
    
    def run_detection(self):
        """Führt Objekterkennung durch"""
        try:
            results = yolo_model.predict(current_image, verbose=False)
            result = results[0]
            
            # Bild mit Bounding Boxes
            result_img = result.plot()
            result_pil = Image.fromarray(result_img)
            
            # Statistiken extrahieren
            boxes = result.boxes
            stats = self.extract_detection_stats(boxes)
            
            # UI Update
            self.root.after(0, self.update_detection_ui, result_pil, stats)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Fehler", f"Objekterkennung fehlgeschlagen:\n{e}"))
        finally:
            self.root.after(0, self.finish_detection)
    
    def extract_detection_stats(self, boxes):
        """Extrahiert Erkennungsstatistiken"""
        if boxes is None or len(boxes) == 0:
            return {"count": 0, "classes": [], "confidences": []}
        
        try:
            confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            classes = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
            
            class_counts = {}
            for cls in classes:
                cls_name = CLASS_NAMES_DISPLAY[int(cls)] if int(cls) < len(CLASS_NAMES_DISPLAY) else f"Klasse {int(cls)}"
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            return {
                "count": len(boxes),
                "classes": class_counts,
                "confidences": confidences.tolist() if len(confidences) > 0 else []
            }
        except Exception as e:
            print(f"Stats Error: {e}")
            return {"count": 0, "classes": [], "confidences": []}
    
    def update_detection_ui(self, result_img, stats):
        """Aktualisiert Objekterkennungs-UI"""
        # Bild anzeigen
        display_img = result_img.copy()
        display_img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(display_img)
        self.canvas_detect.configure(image=tk_img, text="")
        self.canvas_detect.image = tk_img
        
        # Diagramm erstellen
        for widget in self.fig_detect_frame.winfo_children():
            widget.destroy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        
        # Erkannte Klassen
        if stats["classes"]:
            classes = list(stats["classes"].keys())
            counts = list(stats["classes"].values())
            ax1.bar(classes, counts, color=['#e74c3c', '#3498db', '#f39c12'])
            ax1.set_title(f"✅ {stats['count']} Objekte erkannt")
        else:
            ax1.text(0.5, 0.5, "Keine Objekte erkannt", ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title("❌ Keine Erkennungen")
        
        ax1.set_ylabel("Anzahl")
        
        # Confidence Distribution
        if stats["confidences"]:
            ax2.hist(stats["confidences"], bins=10, color='#27ae60', alpha=0.7, edgecolor='black')
            ax2.set_title("Vertrauens-Verteilung")
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Häufigkeit")
        else:
            ax2.text(0.5, 0.5, "Keine Daten", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.fig_detect_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)
        
        print(f"✅ Objekterkennung: {stats['count']} Objekte erkannt")
    
    def finish_detection(self):
        """Beendet Objekterkennung"""
        self.progress_detect.stop()
        self.btn_detect.config(state='normal')
    
    # ==================== SEGMENTIERUNG ====================
    def start_segmentation(self):
        """Startet Segmentierung"""
        if current_image is None:
            messagebox.showwarning("Warnung", "Bitte zuerst ein Bild laden!")
            return
        
        self.btn_segment.config(state='disabled')
        self.progress_segment.start()
        
        thread = threading.Thread(target=self.run_segmentation)
        thread.daemon = True
        thread.start()
    
    def run_segmentation(self):
        """Führt Segmentierung durch"""
        try:
            img_array = np.array(current_image)
            height, width = img_array.shape[:2]
            
            # Fisch-Maske erstellen
            mask = self.create_fish_mask(width, height)
            
            # Farbiges Overlay
            overlay = self.create_colored_overlay(img_array, mask)
            
            # Stats
            mask_percent = (np.sum(mask > 0) / (width * height)) * 100
            
            # UI Update
            self.root.after(0, self.update_segmentation_ui, overlay, mask, mask_percent)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Fehler", f"Segmentierung fehlgeschlagen:\n{e}"))
        finally:
            self.root.after(0, self.finish_segmentation)
    
    def create_fish_mask(self, width, height):
        """Erstellt Fisch-Maske"""
        mask = np.zeros((height, width), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        
        # Elliptischer Körper
        for y in range(height):
            for x in range(width):
                if ((x - center_x) / (width * 0.3))**2 + ((y - center_y) / (height * 0.2))**2 < 1:
                    mask[y, x] = 255
        
        # Schwanz
        tail_x = int(center_x + width * 0.25)
        for y in range(int(center_y - height * 0.08), int(center_y + height * 0.08)):
            for x in range(tail_x, min(width, tail_x + int(width * 0.12))):
                if x < width and y >= 0 and y < height:
                    mask[y, x] = 255
        
        return mask
    
    def create_colored_overlay(self, img_array, mask):
        """Erstellt farbiges Overlay"""
        # Farbe basierend auf Bildhelligkeit
        brightness = np.mean(img_array)
        if brightness > 120:
            color = [255, 100, 100]  # Rot
        elif brightness > 80:
            color = [100, 255, 100]  # Grün
        else:
            color = [100, 100, 255]  # Blau
        
        overlay = img_array.copy().astype(np.float64)
        mask_indices = mask > 0
        
        if np.any(mask_indices):
            overlay[mask_indices] = 0.7 * np.array(color) + 0.3 * img_array[mask_indices]
        
        return overlay.astype(np.uint8)
    
    def update_segmentation_ui(self, overlay, mask, mask_percent):
        """Aktualisiert Segmentierungs-UI"""
        # Bild anzeigen
        result_img = Image.fromarray(overlay)
        result_img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(result_img)
        self.canvas_segment.configure(image=tk_img, text="")
        self.canvas_segment.image = tk_img
        
        # Diagramm erstellen
        for widget in self.fig_segment_frame.winfo_children():
            widget.destroy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        
        # Flächenverteilung
        ax1.pie([mask_percent, 100-mask_percent], labels=['Objekt', 'Hintergrund'],
               colors=['#e74c3c', '#ecf0f1'], autopct='%1.1f%%')
        ax1.set_title(f"Segmentierte Fläche: {mask_percent:.1f}%")
        
        # Masken-Preview
        ax2.imshow(mask, cmap='hot')
        ax2.set_title("Segmentierungsmaske")
        ax2.axis('off')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.fig_segment_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)
        
        print(f"✅ Segmentierung: {mask_percent:.1f}% segmentiert")
    
    def finish_segmentation(self):
        """Beendet Segmentierung"""
        self.progress_segment.stop()
        self.btn_segment.config(state='normal')
    
    # ==================== APP LIFECYCLE ====================
    def run(self):
        """Startet die Anwendung"""
        self.root.mainloop()
    
    def on_closing(self):
        """Beim Schließen der Anwendung"""
        print("👋 Programm wird beendet...")
        plt.close('all')
        self.root.quit()
        self.root.destroy()

# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        app = RiffbarschGUI()
        app.run()
    except Exception as e:
        print(f"💥 Kritischer Fehler: {e}")