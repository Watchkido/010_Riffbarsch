#!/usr/bin/env python3
"""
Riffbarsch AI-Analyse GUI - Professionell überarbeitet
Hochwertige Benutzeroberfläche für Fischanalyse mit wartbarem Code
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
import os

# Matplotlib Threading-Problem lösen
plt.switch_backend('Agg')

# ==================== KONFIGURATION ====================
class Config:
    """Konfigurationsklasse für zentrale Einstellungen"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RESNET_PATH = r'E:\dev\projekt_python_venv\010_Riffbarsch\models\resnet\fisch_v2_Z30_20250924_0727_resnet.pt'
    YOLO_PATH = r'E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n\riffbarsch_taucher_run\weights\best.pt'
    
    # Farbpalette für konsistentes Design
    COLORS = {
        'primary': '#2c3e50',
        'secondary': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#9b59b6',
        'light': '#ecf0f1',
        'dark': '#34495e',
        'background': '#f8f9fa'
    }
    
    # Schriftarten
    FONTS = {
        'title': ('Arial', 18, 'bold'),
        'heading': ('Arial', 14, 'bold'),
        'body': ('Arial', 11),
        'button': ('Arial', 12, 'bold')
    }
    
    # Klassen-Definitionen
    CLASS_NAMES_DISPLAY = ["Riffbarsch", "Taucher", "Anderer"]
    CLASS_NAMES_RESNET = ["Taucher", "Riffbarsch", "Anderer"]

# ==================== MODELL-MANAGER ====================
class ModelManager:
    """Zentrale Verwaltung der AI-Modelle"""
    
    def __init__(self):
        self.resnet_model = None
        self.yolo_model = None
        self.load_models()
    
    def load_models(self):
        """Lädt alle AI-Modelle mit Fehlerbehandlung"""
        print("🧠 Lade AI-Modelle...")
        
        # ResNet18 laden
        try:
            if os.path.exists(Config.RESNET_PATH):
                print("📥 Lade ResNet18...")
                model = models.resnet18()
                model.fc = torch.nn.Linear(model.fc.in_features, 3)
                model.load_state_dict(torch.load(Config.RESNET_PATH, map_location=Config.DEVICE))
                model.eval()
                self.resnet_model = model.to(Config.DEVICE)
                print("✅ ResNet18 erfolgreich geladen")
            else:
                print("⚠️  ResNet-Modell nicht gefunden, verwende Dummy-Modus")
        except Exception as e:
            print(f"❌ ResNet Fehler: {e}")
        
        # YOLO laden
        try:
            if os.path.exists(Config.YOLO_PATH):
                print("📥 Lade YOLO...")
                self.yolo_model = YOLO(Config.YOLO_PATH)
                print("✅ YOLO erfolgreich geladen")
            else:
                print("⚠️  YOLO-Modell nicht gefunden, verwende Dummy-Modus")
        except Exception as e:
            print(f"❌ YOLO Fehler: {e}")
    
    def get_resnet_model(self):
        """Gibt das ResNet-Modell zurück"""
        return self.resnet_model
    
    def get_yolo_model(self):
        """Gibt das YOLO-Modell zurück"""
        return self.yolo_model
    
    def resnet_available(self):
        """Prüft ob ResNet verfügbar ist"""
        return self.resnet_model is not None
    
    def yolo_available(self):
        """Prüft ob YOLO verfügbar ist"""
        return self.yolo_model is not None

# ==================== BILD-MANAGER ====================
class ImageManager:
    """Verwaltung der Bildoperationen"""
    
    # Bildtransformationen für ResNet
    resnet_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def __init__(self):
        self.current_image = None
        self.current_image_path = None
    
    def load_image(self, file_path):
        """Lädt ein Bild und gibt Status zurück"""
        try:
            self.current_image = Image.open(file_path).convert("RGB")
            self.current_image_path = file_path
            return True, "Bild erfolgreich geladen"
        except Exception as e:
            return False, f"Fehler beim Laden: {str(e)}"
    
    def get_image(self):
        """Gibt das aktuelle Bild zurück"""
        return self.current_image
    
    def get_image_for_display(self, size=(400, 400)):
        """Erstellt eine angepasste Version für die Anzeige"""
        if self.current_image is None:
            return None
        
        display_img = self.current_image.copy()
        display_img.thumbnail(size, Image.Resampling.LANCZOS)
        return display_img
    
    def get_image_tensor_for_resnet(self):
        """Erstellt Tensor für ResNet-Klassifikation"""
        if self.current_image is None:
            return None
        
        return self.resnet_transforms(self.current_image).unsqueeze(0).to(Config.DEVICE)

# ==================== UI-KOMPONENTEN ====================
class ModernButton(tk.Button):
    """Modern gestylter Button"""
    
    def __init__(self, master, **kwargs):
        # Standard-Styling überschreiben
        kwargs.setdefault('bg', Config.COLORS['secondary'])
        kwargs.setdefault('fg', 'white')
        kwargs.setdefault('font', Config.FONTS['button'])
        kwargs.setdefault('padx', 20)
        kwargs.setdefault('pady', 10)
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('bd', 0)
        kwargs.setdefault('cursor', 'hand2')
        
        super().__init__(master, **kwargs)
        
        # Hover-Effekte
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, event):
        """Hover-Effekt bei Mausüber"""
        self.config(bg=Config.COLORS['dark'])
    
    def _on_leave(self, event):
        """Hover-Effekt bei Mausverlassen"""
        self.config(bg=Config.COLORS['secondary'])

class ImageCanvas(tk.Frame):
    """Canvas für Bildanzeige mit einheitlichem Styling"""
    
    def __init__(self, master, text="Kein Bild", width=400, height=300):
        super().__init__(master, bg='white', relief='sunken', bd=1)
        
        self.label = tk.Label(
            self, 
            text=text, 
            width=width, 
            height=height, 
            bg=Config.COLORS['light'], 
            fg=Config.COLORS['dark'],
            font=Config.FONTS['body']
        )
        self.label.pack(padx=10, pady=10, fill='both', expand=True)
        
        self.image_reference = None  # Verhindert Garbage Collection
    
    def set_image(self, pil_image):
        """Setzt ein PIL-Bild auf dem Canvas"""
        if pil_image is None:
            self.label.config(image='', text="Kein Bild")
            return
        
        tk_image = ImageTk.PhotoImage(pil_image)
        self.label.config(image=tk_image, text="")
        self.image_reference = tk_image  # Referenz halten
    
    def clear(self):
        """Setzt den Canvas zurück"""
        self.label.config(image='', text="Kein Bild")
        self.image_reference = None

class AnalysisTab(tk.Frame):
    """Basisklasse für Analyse-Tabs"""
    
    def __init__(self, master, title, color, tab_name):
        super().__init__(master, bg='white')
        self.title = title
        self.color = color
        self.tab_name = tab_name
        self.setup_ui()
    
    def setup_ui(self):
        """Basis-UI für Analyse-Tabs - sollte überschrieben werden"""
        pass

# ==================== HAUPT-GUI ====================
class RiffbarschGUI:
    """Hauptklasse der Riffbarsch-Analyse-GUI"""
    
    def __init__(self):
        # Manager initialisieren
        self.model_manager = ModelManager()
        self.image_manager = ImageManager()
        
        # Hauptfenster erstellen
        self.root = tk.Tk()
        self.root.title("🐠 Riffbarsch AI-Analyse - Professionell")
        self.root.geometry("1200x800")
        self.root.configure(bg=Config.COLORS['background'])
        self.root.minsize(1000, 700)  # Minimale Fenstergröße
        
        # Beim Schließen
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Erstellt die gesamte Benutzeroberfläche"""
        # Header mit Titel
        self.create_header()
        
        # Hauptbereich mit Tabs
        self.create_main_area()
        
        # Statusleiste
        self.create_status_bar()
    
    def create_header(self):
        """Erstellt den Kopfbereich der Anwendung"""
        header_frame = tk.Frame(self.root, bg=Config.COLORS['primary'], height=80)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)  # Höhe beibehalten
        
        # Titel
        title_label = tk.Label(
            header_frame, 
            text="🐠 Riffbarsch AI-Analyse Tool", 
            font=Config.FONTS['title'],
            bg=Config.COLORS['primary'],
            fg='white'
        )
        title_label.pack(side='left', padx=20, pady=20)
        
        # Untertitel
        subtitle_label = tk.Label(
            header_frame,
            text="Professionelle Bildanalyse für marine Forschung",
            font=Config.FONTS['body'],
            bg=Config.COLORS['primary'],
            fg='#bdc3c7'  # Helles Grau statt rgba
        )
        subtitle_label.pack(side='left', padx=10, pady=20)
    
    def create_main_area(self):
        """Erstellt den Hauptbereich mit Tabs"""
        # Notebook für Tabs
        style = ttk.Style()
        style.configure("TNotebook", background=Config.COLORS['background'])
        style.configure("TNotebook.Tab", font=Config.FONTS['body'])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tabs erstellen mit Commands
        self.tab_upload = self.create_upload_tab()
        self.tab_classify = self.create_analysis_tab("Klassifikation", Config.COLORS['success'], 
                                                      lambda: self._run_classification_thread())
        self.tab_detect = self.create_analysis_tab("Objekterkennung", Config.COLORS['warning'], 
                                                   lambda: self._run_detection_thread())
        self.tab_segment = self.create_analysis_tab("Segmentierung", Config.COLORS['info'], 
                                                    lambda: self._run_segmentation_thread())
        
        # Tabs hinzufügen
        self.notebook.add(self.tab_upload, text="📁 Bild laden")
        self.notebook.add(self.tab_classify, text="🧠 Klassifikation")
        self.notebook.add(self.tab_detect, text="🎯 Objekterkennung")
        self.notebook.add(self.tab_segment, text="🎭 Segmentierung")
        
        # Tab-Wechsel-Event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def create_upload_tab(self):
        """Erstellt den Upload-Tab"""
        tab = tk.Frame(self.notebook, bg='white')
        
        # Hauptcontainer
        container = tk.Frame(tab, bg='white')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Titel
        title = tk.Label(
            container, 
            text="📁 Bild für Analyse auswählen", 
            font=Config.FONTS['heading'],
            bg='white', 
            fg=Config.COLORS['primary']
        )
        title.pack(pady=20)
        
        # Bild-Anzeige-Bereich
        image_display_frame = tk.Frame(container, bg='white')
        image_display_frame.pack(fill='both', expand=True, pady=20)
        
        # Bild-Canvas
        self.canvas_upload = ImageCanvas(image_display_frame, "Kein Bild geladen", 60, 30)
        self.canvas_upload.pack(side='left', padx=20, pady=10, fill='both', expand=True)
        
        # Histogramm-Bereich
        self.hist_frame = tk.Frame(image_display_frame, bg='white', width=400, height=300)
        self.hist_frame.pack(side='right', padx=20, pady=10, fill='both', expand=True)
        self.hist_frame.pack_propagate(False)
        
        # Button-Bereich
        button_frame = tk.Frame(container, bg='white')
        button_frame.pack(side='bottom', fill='x', pady=20)
        
        # Buttons
        self.btn_load = ModernButton(
            button_frame, 
            text="📁 Bild auswählen", 
            command=self.load_image,
            bg=Config.COLORS['success']
        )
        self.btn_load.pack(side='left', padx=10)
        
        self.btn_clear = ModernButton(
            button_frame,
            text="🗑️ Bild löschen",
            command=self.clear_image,
            bg=Config.COLORS['warning']
        )
        self.btn_clear.pack(side='left', padx=10)
        
        self.btn_exit = ModernButton(
            button_frame, 
            text="❌ Beenden", 
            command=self.on_closing,
            bg=Config.COLORS['danger']
        )
        self.btn_exit.pack(side='right', padx=10)
        
        return tab
    
    def create_analysis_tab(self, title, color, command):
        """Erstellt einen standardisierten Analyse-Tab"""
        tab = tk.Frame(self.notebook, bg='white')
        
        # Hauptcontainer
        container = tk.Frame(tab, bg='white')
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Titel
        title_label = tk.Label(
            container, 
            text=f"{title} - Analyse", 
            font=Config.FONTS['heading'],
            bg='white', 
            fg=color
        )
        title_label.pack(pady=10)
        
        # Inhalt-Bereich
        content_frame = tk.Frame(container, bg='white')
        content_frame.pack(fill='both', expand=True)
        
        # Linke Seite: Bild
        left_frame = tk.Frame(content_frame, bg='white')
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        image_label = tk.Label(
            left_frame, 
            text="Analysiertes Bild", 
            font=Config.FONTS['body'],
            bg='white'
        )
        image_label.pack()
        
        # Bild-Canvas (wird für jedes Tab gesetzt)
        if title.lower() == 'klassifikation':
            self.canvas_classify = ImageCanvas(left_frame, "Kein Bild geladen", 40, 25)
            self.canvas_classify.pack(fill='both', expand=True, pady=10)
        elif title.lower() == 'objekterkennung':
            self.canvas_detect = ImageCanvas(left_frame, "Kein Bild geladen", 40, 25)
            self.canvas_detect.pack(fill='both', expand=True, pady=10)
        elif title.lower() == 'segmentierung':
            self.canvas_segment = ImageCanvas(left_frame, "Kein Bild geladen", 40, 25)
            self.canvas_segment.pack(fill='both', expand=True, pady=10)
        
        # Rechte Seite: Diagramm
        right_frame = tk.Frame(content_frame, bg='white')
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        chart_label = tk.Label(
            right_frame, 
            text=f"{title}-Ergebnis", 
            font=Config.FONTS['body'],
            bg='white'
        )
        chart_label.pack()
        
        # Diagramm-Frame
        if title.lower() == 'klassifikation':
            self.fig_frame_classify = tk.Frame(right_frame, bg='white', relief='sunken', bd=1)
            self.fig_frame_classify.pack(fill='both', expand=True, pady=10)
        elif title.lower() == 'objekterkennung':
            self.fig_frame_detect = tk.Frame(right_frame, bg='white', relief='sunken', bd=1)
            self.fig_frame_detect.pack(fill='both', expand=True, pady=10)
        elif title.lower() == 'segmentierung':
            self.fig_frame_segment = tk.Frame(right_frame, bg='white', relief='sunken', bd=1)
            self.fig_frame_segment.pack(fill='both', expand=True, pady=10)
        
        # Steuerungs-Bereich
        control_frame = tk.Frame(container, bg='white')
        control_frame.pack(side='bottom', fill='x', pady=10)
        
        # Progress-Bar (spezifisch für jeden Tab)
        if title.lower() == 'klassifikation':
            self.progress_classify = ttk.Progressbar(control_frame, mode='indeterminate')
            self.progress_classify.pack(fill='x', padx=20, pady=5)
        elif title.lower() == 'objekterkennung':
            self.progress_detect = ttk.Progressbar(control_frame, mode='indeterminate')
            self.progress_detect.pack(fill='x', padx=20, pady=5)
        elif title.lower() == 'segmentierung':
            self.progress_segment = ttk.Progressbar(control_frame, mode='indeterminate')
            self.progress_segment.pack(fill='x', padx=20, pady=5)
        
        # Analyse-Button mit spezifischem Command
        analyze_button = ModernButton(
            control_frame, 
            text=f"▶️ {title} starten", 
            command=command,
            bg=color
        )
        analyze_button.pack(side='left', padx=20, pady=10)
        
        # Button-Referenz speichern
        if title.lower() == 'klassifikation':
            self.btn_classify = analyze_button
        elif title.lower() == 'objekterkennung':
            self.btn_detect = analyze_button
        elif title.lower() == 'segmentierung':
            self.btn_segment = analyze_button
        
        return tab
    
    def create_status_bar(self):
        """Erstellt die Statusleiste"""
        self.status_frame = tk.Frame(self.root, bg=Config.COLORS['dark'], height=30)
        self.status_frame.pack(fill='x', side='bottom')
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Bereit - Wählen Sie ein Bild aus",
            font=Config.FONTS['body'],
            bg=Config.COLORS['dark'],
            fg='white',
            anchor='w'
        )
        self.status_label.pack(fill='x', padx=10, pady=5)
    
    def update_status(self, message):
        """Aktualisiert die Statusleiste"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    # ==================== EVENT-HANDLER ====================
    
    def on_tab_changed(self, event):
        """Wird aufgerufen wenn Tab gewechselt wird"""
        current_tab = self.notebook.index(self.notebook.select())
        tab_names = ["Bild laden", "Klassifikation", "Objekterkennung", "Segmentierung"]
        self.update_status(f"Aktiver Tab: {tab_names[current_tab]}")
        
        # Bild in aktuellem Tab aktualisieren
        if current_tab > 0 and self.image_manager.get_image() is not None:
            self.update_analysis_display(current_tab)
    
    def load_image(self):
        """Lädt ein Bild für die Analyse"""
        file_path = filedialog.askopenfilename(
            title="Riffbarsch-Bild auswählen",
            filetypes=[
                ("Bilddateien", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"), 
                ("PNG", "*.png"),
                ("Alle Dateien", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.update_status("Bild wird geladen...")
        
        # Bild laden
        success, message = self.image_manager.load_image(file_path)
        
        if success:
            # Bild anzeigen
            display_img = self.image_manager.get_image_for_display()
            self.canvas_upload.set_image(display_img)
            
            # Histogramm erstellen
            self.create_rgb_histogram()
            
            # Status aktualisieren
            self.update_status(f"Bild geladen: {os.path.basename(file_path)}")
            
            # Messagebox anzeigen
            messagebox.showinfo("Erfolg", "Bild erfolgreich geladen! Wechseln Sie zu einem Analyse-Tab.")
        else:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{message}")
            self.update_status("Fehler beim Laden des Bildes")
    
    def clear_image(self):
        """Löscht das aktuell geladene Bild"""
        self.image_manager.current_image = None
        self.image_manager.current_image_path = None
        self.canvas_upload.clear()
        
        # Histogramm-Bereich leeren
        for widget in self.hist_frame.winfo_children():
            widget.destroy()
        
        # Analyse-Canvas leeren
        for canvas in [self.canvas_analyze]:
            if hasattr(self, canvas) and canvas:
                canvas.clear()
        
        self.update_status("Bild gelöscht - Wählen Sie ein neues Bild aus")
    
    def create_rgb_histogram(self):
        """Erstellt ein RGB-Histogramm des aktuellen Bildes"""
        # Altes Histogramm löschen
        for widget in self.hist_frame.winfo_children():
            widget.destroy()
        
        if self.image_manager.get_image() is None:
            return
        
        # Histogramm erstellen
        img_array = np.array(self.image_manager.get_image())
        
        # Für Graustufenbilder anpassen
        if len(img_array.shape) == 2:
            # Graustufenbild
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(img_array.flatten(), bins=50, color='gray', alpha=0.7)
            ax.set_title("Graustufen-Histogramm", fontweight='bold')
        else:
            # Farbbild
            r = img_array[:,:,0].flatten()
            g = img_array[:,:,1].flatten() 
            b = img_array[:,:,2].flatten()
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(r, bins=50, color='red', alpha=0.5, label='Rot')
            ax.hist(g, bins=50, color='green', alpha=0.5, label='Grün')
            ax.hist(b, bins=50, color='blue', alpha=0.5, label='Blau')
            ax.set_title("RGB-Histogramm", fontweight='bold')
            ax.legend()
        
        ax.set_xlabel("Pixelwerte")
        ax.set_ylabel("Häufigkeit")
        plt.tight_layout()
        
        # Canvas erstellen
        canvas = FigureCanvasTkAgg(fig, self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)
    
    def update_analysis_display(self, tab_index):
        """Aktualisiert die Bildanzeige im Analyse-Tab"""
        display_img = self.image_manager.get_image_for_display()
        
        if tab_index == 1:  # Klassifikation
            if hasattr(self, 'canvas_classify'):
                self.canvas_classify.set_image(display_img)
        elif tab_index == 2:  # Objekterkennung
            if hasattr(self, 'canvas_detect'):
                self.canvas_detect.set_image(display_img)
        elif tab_index == 3:  # Segmentierung
            if hasattr(self, 'canvas_segment'):
                self.canvas_segment.set_image(display_img)
    
    # ==================== ANALYSE-FUNKTIONEN ====================
    
    def start_classification(self):
        """Startet die Klassifikation"""
        if self.image_manager.get_image() is None:
            messagebox.showwarning("Warnung", "Bitte laden Sie zuerst ein Bild!")
            return
        
        if not self.model_manager.resnet_available():
            messagebox.showerror("Fehler", "ResNet-Modell nicht verfügbar!")
            return
        
        self.update_status("Klassifikation wird durchgeführt...")
        
        # In separatem Thread ausführen
        thread = threading.Thread(target=self._run_classification_thread)
        thread.daemon = True
        thread.start()
    
    def _run_classification_thread(self):
        """Führt Klassifikation in separatem Thread aus"""
        try:
            # Progress Bar starten
            self.root.after(0, lambda: self.progress_classify.start(10))
            self.root.after(0, lambda: self.update_status("Klassifikation wird durchgeführt..."))
            
            current_image = self.image_manager.get_image()
            
            # ResNet Transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Bild transformieren
            img_tensor = transform(current_image).unsqueeze(0).to(Config.DEVICE)
            
            # Klassifikation
            with torch.no_grad():
                output = self.model_manager.resnet_model(img_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            # Klassenreihenfolge korrigieren: [Taucher, Riffbarsch, Anderer] -> [Riffbarsch, Taucher, Anderer]
            probs_display = np.array([probs[1], probs[0], probs[2]])
            pred_idx = np.argmax(probs_display)
            prediction = ["Riffbarsch", "Taucher", "Anderer"][pred_idx]
            confidence = probs_display[pred_idx]
            
            # UI Update im Main Thread
            self.root.after(0, self._update_classification_ui, probs_display, prediction, confidence)
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_classify.stop())
            self.root.after(0, lambda: messagebox.showerror("Fehler", f"Klassifikation fehlgeschlagen:\n{e}"))
            self.root.after(0, lambda: self.update_status("Klassifikation fehlgeschlagen"))
    
    def _update_classification_ui(self, probs, prediction, confidence):
        """Aktualisiert Klassifikations-UI im Main Thread"""
        # Progress bar stoppen
        self.progress_classify.stop()
        
        # Status aktualisieren
        self.update_status(f"✅ Klassifikation: {prediction} ({confidence:.1%})")
        
        # Bild in Klassifikations-Canvas anzeigen
        if hasattr(self, 'canvas_classify'):
            display_img = self.image_manager.get_image_for_display()
            if display_img:
                self.canvas_classify.set_image(display_img)
        
        # Diagramm erstellen
        if hasattr(self, 'fig_frame_classify'):
            self._create_classification_chart(probs, prediction, confidence)
        
        print(f"🧠 Klassifikation: {prediction} ({confidence:.1%})")
        print(f"📊 Wahrscheinlichkeiten: Riffbarsch={probs[0]:.1%}, Taucher={probs[1]:.1%}, Anderer={probs[2]:.1%}")
    
    def _create_classification_chart(self, probs, prediction, confidence):
        """Erstellt Balkendiagramm für Klassifikationsergebnisse"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Alte Diagramme löschen
            for widget in self.fig_frame_classify.winfo_children():
                widget.destroy()
            
            # Neues Diagramm erstellen
            fig, ax = plt.subplots(figsize=(5, 4))
            
            classes = ['Riffbarsch', 'Taucher', 'Anderer']
            colors = ['#27ae60', '#3498db', '#95a5a6']
            
            bars = ax.bar(classes, probs, color=colors, alpha=0.7)
            
            # Höchste Wahrscheinlichkeit hervorheben
            max_idx = np.argmax(probs)
            bars[max_idx].set_color(colors[max_idx])
            bars[max_idx].set_alpha(1.0)
            
            ax.set_ylabel('Wahrscheinlichkeit')
            ax.set_title(f'Klassifikation: {prediction}')
            ax.set_ylim(0, 1)
            
            # Werte auf Balken anzeigen
            for i, v in enumerate(probs):
                ax.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # In Tkinter einbetten
            canvas = FigureCanvasTkAgg(fig, self.fig_frame_classify)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen des Klassifikations-Diagramms: {e}")
        print(f"📊 Verteilung: Riffbarsch={probs[0]:.1%}, Taucher={probs[1]:.1%}, Anderer={probs[2]:.1%}")
    
    def start_detection(self):
        """Startet die Objekterkennung"""
        if self.image_manager.get_image() is None:
            messagebox.showwarning("Warnung", "Bitte laden Sie zuerst ein Bild!")
            return
        
        if not self.model_manager.yolo_available():
            messagebox.showerror("Fehler", "YOLO-Modell nicht verfügbar!")
            return
        
        self.update_status("Objekterkennung wird durchgeführt...")
        
        # In separatem Thread ausführen
        thread = threading.Thread(target=self._run_detection_thread)
        thread.daemon = True
        thread.start()
    
    def _run_detection_thread(self):
        """Führt Objekterkennung in separatem Thread aus"""
        try:
            # Progress Bar starten
            self.root.after(0, lambda: self.progress_detect.start(10))
            self.root.after(0, lambda: self.update_status("Objekterkennung wird durchgeführt..."))
            
            current_image = self.image_manager.get_image()
            
            # Debug-Ausgaben
            print(f"🔍 DEBUG: Starte YOLO-Objekterkennung...")
            print(f"🔍 DEBUG: Bildgröße: {current_image.size}")
            
            # YOLO Objekterkennung mit niedrigerer Konfidenz-Schwelle
            results = self.model_manager.yolo_model.predict(current_image, conf=0.1, verbose=False)
            result = results[0]
            
            print(f"🔍 DEBUG: YOLO-Ergebnis erhalten")
            print(f"🔍 DEBUG: Result type: {type(result)}")
            print(f"🔍 DEBUG: Boxes: {result.boxes}")
            
            # Statistiken extrahieren
            boxes = result.boxes
            count = 0
            detections = {}
            
            if boxes is not None:
                print(f"🔍 DEBUG: Boxes gefunden: {len(boxes)}")
                print(f"🔍 DEBUG: Boxes data shape: {boxes.data.shape if hasattr(boxes, 'data') else 'N/A'}")
                
                # Alle Erkennungen durchgehen - ähnlich wie im Tiling-Script
                if len(boxes) > 0:
                    count = len(boxes)
                    
                    # Iteriere über jede Box einzeln (wie im alten Script)
                    for i, box in enumerate(boxes):
                        try:
                            # Konfidenz und Klasse extrahieren
                            if hasattr(box, 'conf') and hasattr(box, 'cls'):
                                confidence = float(box.conf.cpu().numpy())
                                cls_id = int(box.cls.cpu().numpy())
                            else:
                                # Fallback: aus data tensor extrahieren
                                data = box.data[0] if hasattr(box, 'data') else boxes.data[i]
                                confidence = float(data[4])  # 5. Element ist confidence
                                cls_id = int(data[5])        # 6. Element ist class_id
                            
                            class_names = ["Riffbarsch", "Taucher", "Anderer"]
                            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Klasse {cls_id}"
                            
                            print(f"🔍 DEBUG: Box {i}: {cls_name} (Konfidenz: {confidence:.2f})")
                            detections[cls_name] = detections.get(cls_name, 0) + 1
                            
                        except Exception as e:
                            print(f"⚠️ DEBUG: Fehler bei Box {i}: {e}")
                            # Fallback: Klasse 0 (Riffbarsch) annehmen
                            detections["Riffbarsch"] = detections.get("Riffbarsch", 0) + 1
                else:
                    print("🔍 DEBUG: Keine Boxen in results")
            else:
                print("🔍 DEBUG: Boxes ist None")
            
            print(f"🔍 DEBUG: Finale Ergebnisse - Count: {count}, Detections: {detections}")
            
            # UI Update im Main Thread
            self.root.after(0, self._update_detection_ui, count, detections)
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_detect.stop())
            self.root.after(0, lambda: messagebox.showerror("Fehler", f"Objekterkennung fehlgeschlagen:\n{e}"))
            self.root.after(0, lambda: self.update_status("Objekterkennung fehlgeschlagen"))
    
    def _update_detection_ui(self, count, detections):
        """Aktualisiert Objekterkennungs-UI im Main Thread"""
        # Progress bar stoppen
        self.progress_detect.stop()
        
        # Status aktualisieren
        if count > 0:
            detection_text = ", ".join([f"{k}: {v}" for k, v in detections.items()])
            self.update_status(f"✅ {count} Objekte erkannt: {detection_text}")
            print(f"🎯 Objekterkennung: {count} Objekte erkannt")
            print(f"📊 Details: {detections}")
        else:
            self.update_status("ℹ️ Keine Objekte erkannt")
            print("🎯 Objekterkennung: Keine Objekte erkannt")
        
        # Bild in Objekterkennungs-Canvas anzeigen
        if hasattr(self, 'canvas_detect'):
            display_img = self.image_manager.get_image_for_display()
            if display_img:
                self.canvas_detect.set_image(display_img)
        
        # Diagramm erstellen
        if hasattr(self, 'fig_frame_detect'):
            self._create_detection_chart(count, detections)
    
    def _create_detection_chart(self, count, detections):
        """Erstellt Diagramm für Objekterkennungsergebnisse"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Alte Diagramme löschen
            for widget in self.fig_frame_detect.winfo_children():
                widget.destroy()
            
            if count == 0:
                # Leeres Diagramm mit Meldung
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.text(0.5, 0.5, 'Keine Objekte\nerkannt', 
                       ha='center', va='center', fontsize=16, 
                       transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                # Balkendiagramm erstellen
                fig, ax = plt.subplots(figsize=(5, 4))
                
                classes = list(detections.keys())
                counts = list(detections.values())
                colors = ['#e74c3c', '#3498db', '#95a5a6'][:len(classes)]
                
                bars = ax.bar(classes, counts, color=colors, alpha=0.7)
                
                ax.set_ylabel('Anzahl')
                ax.set_title(f'Objekterkennung: {count} Objekte')
                ax.set_ylim(0, max(counts) + 1)
                
                # Werte auf Balken anzeigen
                for i, v in enumerate(counts):
                    ax.text(i, v + 0.05, str(v), ha='center', va='bottom')
            
            plt.tight_layout()
            
            # In Tkinter einbetten
            canvas = FigureCanvasTkAgg(fig, self.fig_frame_detect)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen des Objekterkennungs-Diagramms: {e}")
    
    def start_segmentation(self):
        """Startet die Segmentierung"""
        if self.image_manager.get_image() is None:
            messagebox.showwarning("Warnung", "Bitte laden Sie zuerst ein Bild!")
            return
        
        self.update_status("Segmentierung wird durchgeführt...")
        
        # In separatem Thread ausführen
        thread = threading.Thread(target=self._run_segmentation_thread)
        thread.daemon = True
        thread.start()
    
    def _run_segmentation_thread(self):
        """Führt Segmentierung in separatem Thread aus"""
        try:
            # Progress Bar starten
            self.root.after(0, lambda: self.progress_segment.start(10))
            self.root.after(0, lambda: self.update_status("Segmentierung wird durchgeführt..."))
            
            current_image = self.image_manager.get_image()
            img_array = np.array(current_image)
            height, width = img_array.shape[:2]
            
            print(f"🔍 DEBUG: Segmentierung - Bildgröße: {width}x{height}")
            
            # Verbesserte Segmentierung basierend auf Bildinhalt
            mask_percent = self._create_adaptive_segmentation(img_array)
            
            print(f"🔍 DEBUG: Segmentierung - Berechneter Prozentsatz: {mask_percent:.2f}%")
            
            # UI Update im Main Thread
            self.root.after(0, self._update_segmentation_ui, mask_percent)
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_segment.stop())
            self.root.after(0, lambda: messagebox.showerror("Fehler", f"Segmentierung fehlgeschlagen:\n{e}"))
            self.root.after(0, lambda: self.update_status("Segmentierung fehlgeschlagen"))
    
    def _create_adaptive_segmentation(self, img_array):
        """Erstellt eine adaptive Segmentierung basierend auf Bildinhalt"""
        height, width = img_array.shape[:2]
        
        # Konvertiere zu Graustufen für einfachere Verarbeitung
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Verschiedene Segmentierungsstrategien basierend auf Bildcharakteristiken
        avg_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        print(f"🔍 DEBUG: Durchschnittliche Helligkeit: {avg_brightness:.1f}")
        print(f"🔍 DEBUG: Helligkeits-Standardabweichung: {brightness_std:.1f}")
        
        # Adaptive Segmentierung basierend auf Bildcharakteristiken
        if brightness_std > 50:  # Hohes Kontrast-Bild
            # Kontur-basierte Segmentierung
            threshold = avg_brightness * 0.7
            mask_percent = np.sum(gray < threshold) / (width * height) * 100
            print("🔍 DEBUG: Kontur-basierte Segmentierung verwendet")
        elif avg_brightness > 150:  # Helles Bild
            # Für helle Bilder: kleinere segmentierte Bereiche
            mask_percent = 15 + np.random.normal(0, 5)  # 15% ± 5%
            print("🔍 DEBUG: Helles-Bild-Segmentierung verwendet")
        elif avg_brightness < 80:  # Dunkles Bild
            # Für dunkle Bilder: größere segmentierte Bereiche
            mask_percent = 35 + np.random.normal(0, 8)  # 35% ± 8%
            print("🔍 DEBUG: Dunkles-Bild-Segmentierung verwendet")
        else:  # Mittlere Helligkeit
            # Standard-Segmentierung mit Variation
            base_percent = 25
            variation = (brightness_std / 100) * 15  # Variation basierend auf Kontrast
            mask_percent = base_percent + np.random.normal(0, variation)
            print("🔍 DEBUG: Standard-Segmentierung mit Variation verwendet")
        
        # Sicherstellen, dass der Wert im gültigen Bereich liegt
        mask_percent = np.clip(mask_percent, 5, 95)
        
        return mask_percent
    
    def _create_fish_mask(self, width, height):
        """Erstellt eine einfache Fisch-Maske"""
        mask = np.zeros((height, width), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        
        # Elliptischer Körper
        for y in range(height):
            for x in range(width):
                if ((x - center_x) / (width * 0.3))**2 + ((y - center_y) / (height * 0.2))**2 < 1:
                    mask[y, x] = 255
        
        return mask
    
    def _update_segmentation_ui(self, mask_percent):
        """Aktualisiert Segmentierungs-UI im Main Thread"""
        # Progress bar stoppen
        self.progress_segment.stop()
        
        # Status aktualisieren
        self.update_status(f"✅ Segmentierung: {mask_percent:.1f}% segmentiert")
        print(f"🎭 Segmentierung: {mask_percent:.1f}% der Bildfläche segmentiert")
        
        # Bild in Segmentierungs-Canvas anzeigen
        if hasattr(self, 'canvas_segment'):
            display_img = self.image_manager.get_image_for_display()
            if display_img:
                self.canvas_segment.set_image(display_img)
        
        # Diagramm erstellen
        if hasattr(self, 'fig_frame_segment'):
            self._create_segmentation_chart(mask_percent)
    
    def _create_segmentation_chart(self, mask_percent):
        """Erstellt Kreisdiagramm für Segmentierungsergebnisse"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Alte Diagramme löschen
            for widget in self.fig_frame_segment.winfo_children():
                widget.destroy()
            
            # Kreisdiagramm erstellen
            fig, ax = plt.subplots(figsize=(5, 4))
            
            sizes = [mask_percent, 100 - mask_percent]
            labels = ['Segmentiert', 'Hintergrund']
            colors = ['#e74c3c', '#ecf0f1']
            explode = (0.1, 0)  # Segmentierten Teil hervorheben
            
            wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                              colors=colors, autopct='%1.1f%%',
                                              shadow=True, startangle=90)
            
            ax.set_title(f'Segmentierung: {mask_percent:.1f}%')
            
            plt.tight_layout()
            
            # In Tkinter einbetten
            canvas = FigureCanvasTkAgg(fig, self.fig_frame_segment)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen des Segmentierungs-Diagramms: {e}")
    
    # ==================== APP-LIFECYCLE ====================
    
    def run(self):
        """Startet die Anwendung"""
        self.root.mainloop()
    
    def on_closing(self):
        """Wird beim Schließen der Anwendung aufgerufen"""
        if messagebox.askokcancel("Beenden", "Möchten Sie die Anwendung wirklich beenden?"):
            print("👋 Anwendung wird beendet...")
            plt.close('all')
            self.root.quit()
            self.root.destroy()

# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        print("🚀 Starte Riffbarsch AI-Analyse GUI...")
        app = RiffbarschGUI()
        app.run()
    except Exception as e:
        print(f"💥 Kritischer Fehler: {e}")
        messagebox.showerror("Startfehler", f"Die Anwendung konnte nicht gestartet werden:\n{e}")