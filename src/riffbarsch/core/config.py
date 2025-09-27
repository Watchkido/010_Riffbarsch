"""
Konfigurationsdatei für die Riffbarsch-Anwendung
Zentrale Verwaltung aller Einstellungen und Pfade
"""

from pathlib import Path
from typing import List, Dict, Any
import os


class Config:
    """Zentrale Konfigurationsklasse"""
    
    # ==================== PFADE ====================
    # Projektpfad
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Modell-Pfade
    RESNET_PATH = MODELS_DIR / "resnet" / "fisch_v2_Z30_20250924_0727_resnet.pt"
    YOLO_PATH = MODELS_DIR / "yolov8n" / "riffbarsch_taucher_run" / "weights" / "best.pt"
    
    # ==================== MODELL-KONFIGURATION ====================
    # Klassen
    NUM_CLASSES = 3
    CLASS_NAMES = ["Riffbarsch", "Taucher", "Anderer"]
    CLASS_COLORS = ["#e74c3c", "#3498db", "#95a5a6"]  # Rot, Blau, Grau
    
    # ResNet-Konfiguration
    RESNET_INPUT_SIZE = (224, 224)
    RESNET_MEANS = [0.485, 0.456, 0.406]
    RESNET_STDS = [0.229, 0.224, 0.225]
    
    # YOLO-Konfiguration  
    YOLO_CONFIDENCE_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.6
    YOLO_INPUT_SIZE = 640
    
    # ==================== GUI-KONFIGURATION ====================
    # Fenster-Einstellungen
    WINDOW_TITLE = "Riffbarsch AI-Analyse"
    WINDOW_SIZE = "1400x800"
    WINDOW_MIN_SIZE = (1200, 700)
    
    # Farben (Modern UI)
    COLORS = {
        'primary': '#2c3e50',      # Dunkelblau
        'secondary': '#34495e',    # Grau-Blau
        'success': '#27ae60',      # Grün
        'warning': '#f39c12',      # Orange
        'danger': '#e74c3c',       # Rot
        'info': '#3498db',         # Blau
        'light': '#ecf0f1',        # Hellgrau
        'dark': '#2c3e50',         # Dunkelgrau
        'white': '#ffffff',
        'background': '#f8f9fa'
    }
    
    # Schriftarten
    FONTS = {
        'title': ('Helvetica', 18, 'bold'),
        'heading': ('Helvetica', 14, 'bold'),
        'body': ('Helvetica', 11),
        'small': ('Helvetica', 9),
        'button': ('Helvetica', 12, 'bold')
    }
    
    # Bildgrößen
    IMAGE_DISPLAY_SIZE = (500, 400)
    THUMBNAIL_SIZE = (150, 150)
    
    # ==================== ANALYSE-KONFIGURATION ====================
    # Segmentierung
    SEGMENT_ELLIPSE_SCALE = (0.35, 0.25)  # Breite, Höhe als Faktor der Bildgröße
    SEGMENT_NOISE_FACTOR = 0.3
    SEGMENT_BRIGHTNESS_THRESHOLD = 0.4
    
    # Threading
    MAX_THREADS = 4
    THREAD_TIMEOUT = 30  # Sekunden
    
    # ==================== LOGGING ====================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "logs" / "riffbarsch.log"
    
    # ==================== VALIDIERUNG ====================
    SUPPORTED_IMAGE_FORMATS = [
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"
    ]
    
    MAX_IMAGE_SIZE = (4000, 4000)  # Maximale Bildgröße
    MIN_IMAGE_SIZE = (100, 100)    # Minimale Bildgröße
    
    @classmethod
    def validate_paths(cls) -> Dict[str, bool]:
        """Validiert alle wichtigen Pfade"""
        validation = {}
        
        # Modell-Pfade prüfen
        validation['resnet_exists'] = cls.RESNET_PATH.exists()
        validation['yolo_exists'] = cls.YOLO_PATH.exists()
        validation['models_dir_exists'] = cls.MODELS_DIR.exists()
        validation['data_dir_exists'] = cls.DATA_DIR.exists()
        
        return validation
    
    @classmethod
    def create_directories(cls):
        """Erstellt notwendige Verzeichnisse"""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed", 
            cls.DATA_DIR / "results",
            cls.PROJECT_ROOT / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Gibt Informationen über verfügbare Hardware zurück"""
        import torch
        
        return {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__
        }