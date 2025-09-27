"""
Modell-Manager für alle AI-Modelle
Verwaltet ResNet18, YOLOv8 und Segmentierungsmodelle
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import Optional, Dict, Any

from .config import Config


class ModelManager:
    """Verwaltet alle AI-Modelle für die Riffbarsch-Analyse"""
    
    def __init__(self):
        """Initialisiert den Model Manager"""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"🔧 Verwende Device: {self.device}")
        
        # Modelle
        self.resnet_model: Optional[nn.Module] = None
        self.yolo_model: Optional[YOLO] = None
        
        # Transformationen
        self.resnet_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Status
        self._models_loaded = False
        
    def load_models(self) -> bool:
        """Lädt alle verfügbaren Modelle"""
        success = True
        
        # ResNet laden
        if not self.load_resnet():
            success = False
            
        # YOLO laden
        if not self.load_yolo():
            success = False
            
        self._models_loaded = success
        return success
    
    def load_resnet(self) -> bool:
        """Lädt das ResNet18-Klassifikationsmodell"""
        try:
            self.logger.info("📥 Lade ResNet18...")
            
            if not Path(Config.RESNET_PATH).exists():
                self.logger.error(f"❌ ResNet-Modell nicht gefunden: {Config.RESNET_PATH}")
                return False
            
            # Modell erstellen
            self.resnet_model = models.resnet18()
            self.resnet_model.fc = nn.Linear(
                self.resnet_model.fc.in_features, 
                Config.NUM_CLASSES
            )
            
            # Gewichte laden
            state_dict = torch.load(Config.RESNET_PATH, map_location=self.device)
            self.resnet_model.load_state_dict(state_dict)
            
            # Auf Device verschieben und in Eval-Modus
            self.resnet_model.to(self.device)
            self.resnet_model.eval()
            
            self.logger.info("✅ ResNet18 erfolgreich geladen")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden von ResNet: {e}")
            self.resnet_model = None
            return False
    
    def load_yolo(self) -> bool:
        """Lädt das YOLOv8-Objekterkennungsmodell"""
        try:
            self.logger.info("📥 Lade YOLOv8...")
            
            if not Path(Config.YOLO_PATH).exists():
                self.logger.error(f"❌ YOLO-Modell nicht gefunden: {Config.YOLO_PATH}")
                return False
            
            # YOLO-Modell laden
            self.yolo_model = YOLO(Config.YOLO_PATH)
            
            self.logger.info("✅ YOLOv8 erfolgreich geladen")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden von YOLO: {e}")
            self.yolo_model = None
            return False
    
    def resnet_available(self) -> bool:
        """Prüft ob ResNet verfügbar ist"""
        return self.resnet_model is not None
    
    def yolo_available(self) -> bool:
        """Prüft ob YOLO verfügbar ist"""
        return self.yolo_model is not None
    
    def models_loaded(self) -> bool:
        """Prüft ob alle Modelle geladen sind"""
        return self._models_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über geladene Modelle zurück"""
        return {
            'device': str(self.device),
            'resnet_loaded': self.resnet_available(),
            'yolo_loaded': self.yolo_available(),
            'all_loaded': self.models_loaded(),
            'num_classes': Config.NUM_CLASSES,
            'class_names': Config.CLASS_NAMES
        }
    
    def cleanup(self):
        """Räumt Ressourcen auf"""
        if self.resnet_model:
            self.resnet_model.cpu()
            self.resnet_model = None
            
        if self.yolo_model:
            self.yolo_model = None
            
        # GPU-Speicher freigeben
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("🧹 Model Manager aufgeräumt")