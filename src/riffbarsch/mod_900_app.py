#!/usr/bin/env python3
"""
Riffbarsch AI Analysis GUI - Verbesserte Struktur
Modulare und wartbare Architektur für Fischanalyse-Anwendung
"""

import tkinter as tk
from tkinter import messagebox
from src.riffbarsch.ui.main_window import RiffbarschMainWindow
from src.riffbarsch.core.model_manager import ModelManager
from src.riffbarsch.core.config import Config
from src.riffbarsch.utils.logging_setup import setup_logging
import logging


class RiffbarschApplication:
    """Hauptanwendungsklasse für die Riffbarsch-Analyse"""
    
    def __init__(self):
        """Initialisiert die Anwendung"""
        self.logger = logging.getLogger(__name__)
        self.root = None
        self.main_window = None
        self.model_manager = None
        
    def initialize(self):
        """Initialisiert alle Komponenten"""
        try:
            # Logging konfigurieren
            setup_logging()
            self.logger.info("🚀 Starte Riffbarsch AI-Analyse...")
            
            # Tkinter Root erstellen
            self.root = tk.Tk()
            self.root.withdraw()  # Verstecken bis alles geladen ist
            
            # Model Manager initialisieren
            self.logger.info("🧠 Lade AI-Modelle...")
            self.model_manager = ModelManager()
            self.model_manager.load_models()
            
            # Hauptfenster erstellen
            self.main_window = RiffbarschMainWindow(
                root=self.root, 
                model_manager=self.model_manager
            )
            
            # Fenster anzeigen
            self.root.deiconify()
            self.logger.info("✅ Anwendung erfolgreich gestartet")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Initialisieren: {e}")
            if self.root:
                messagebox.showerror("Initialisierungsfehler", 
                                   f"Fehler beim Starten der Anwendung:\n{e}")
            return False
    
    def run(self):
        """Startet die Hauptschleife der Anwendung"""
        if not self.initialize():
            return
            
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("👋 Anwendung durch Benutzer beendet")
        except Exception as e:
            self.logger.error(f"❌ Unerwarteter Fehler: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Aufräumen beim Beenden"""
        if self.model_manager:
            self.model_manager.cleanup()
        self.logger.info("🧹 Cleanup abgeschlossen")


def main():
    """Haupteinstiegspunkt der Anwendung"""
    app = RiffbarschApplication()
    app.run()


if __name__ == "__main__":
    main()