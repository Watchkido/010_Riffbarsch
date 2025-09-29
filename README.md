# 🐟 Riffbarsch AI-Analyse

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**KI-gestützte Unterwasser-Fischerkennung und -analyse**

[Installation](#installation) • [Verwendung](#verwendung) • [Features](#features) • [Dokumentation](#dokumentation)

## 📋 Inhaltsverzeichnis

*   [Projektübersicht](#-projektübersicht)
*   [Features](#-features)
*   [Installation](#-installation)
*   [Schnellstart](#-schnellstart)
*   [Verwendung](#-verwendung)
*   [Projektstruktur](#-projektstruktur)
*   [Modelle](#-modelle)
*   [Konfiguration](#-konfiguration)
*   [Entwicklung](#-entwicklung)
*   [Troubleshooting](#-troubleshooting)
*   [Beitragen](#-beitragen)
*   [Lizenz](#-lizenz)
*   [Ressourcen](#-ressourcen)
*   [Autoren & Mitwirkende](#-autoren--mitwirkende)
*   [Roadmap](#-roadmap)

## 🚧 Webseite

Projektwebseite: [https://watchkido.github.io/010_Riffbarsch/](https://watchkido.github.io/010_Riffbarsch/)

## 🌊 Projektübersicht

Das Riffbarsch AI-Analyse Projekt ist eine vollständige Pipeline für die KI-gestützte Erkennung und Analyse von Unterwasserfischen, speziell Riffbarschen. Das System kombiniert moderne Computer Vision Techniken mit einer benutzerfreundlichen GUI.

### 🎯 Hauptziele

*   **Klassifikation**: Unterscheidung zwischen Riffbarsch, Taucher und hart negativ Objekten
*   **Objekterkennung**: Erkennung und Lokalisierung von Fischen in Unterwasserbildern
*   **Segmentierung**: Präzise Abgrenzung von Fischkonturen
*   **coming soon Real-time Analyse**: Schnelle Verarbeitung für praktische Anwendungen

### 🔬 Wissenschaftlicher Hintergrund

Entwickelt für die marine Biodiversitätsforschung, unterstützt das System:

*   Quantitative Fischbestandserfassung
*   Verhaltensanalyse von Meereslebewesen
*   Umweltmonitoring in Korallenriffen
*   Automatisierte Datenauswertung für Meeresbiologen

## ✨ Features

### 🖥️ Benutzeroberfläche

*   GUI mit Tkinter
*   Tab-basierte Navigation (Upload, Klassifikation, Objekterkennung, Segmentierung)
*   Visualisierung der Analyseergebnisse
*   Diagramme mit `matplotlib`

### 🤖 KI-Modelle

*   ResNet18 für Bildklassifikation (3 Klassen)
*   YOLOv8n für Objekterkennung
*   Adaptive Segmentierung basierend auf Bildcharakteristiken
*   RAM-DISC-Beschleunigung 

### 📊 Analysefunktionen

*   Klassenwahrscheinlichkeiten
*   comming soon - Bounding Box Erkennung mit Objektzählung
*   Segmentierungsmasken mit Flächenberechnung
*   Statistische Auswertung der Ergebnisse

### 🔧 Technische Features

*   Robuste Fehlerbehandlung
*   Modulare Architektur für Erweiterbarkeit

## 🚀 Installation

### Voraussetzungen

*   Python 3.8+
*   Software ist für 50GB RAM-DISK ausgelegt
*   4GB+ RAM (TURBO-Module 100GB+ RAM)
*   2GB+ Festplattenspeicher

### 1. Repository klonen

```bash
git clone https://github.com/Watchkido/010_Riffbarsch.git
cd 010_Riffbarsch
```

### 2. Virtual Environment erstellen (empfohlen)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 4. Modelle herunterladen

Stelle sicher, dass die folgenden Modelle verfügbar sind:

*   `models/resnet/fisch_v2_Z30_20250924_0727_resnet.pt`
*   `models/yolov8n/riffbarsch_taucher_run/weights/best.pt`

## ⚡ Schnellstart

### GUI starten

```bash
python src/010_Riffbarsch/main.py
```

### Grundlegende Verwendung

*   **Bild laden**: Tab "Upload" → "Bild laden"
*   **Klassifikation**: Tab "Klassifikation" → "Klassifizieren"
*   **Objekterkennung**: Tab "Objekterkennung" → "Objekte erkennen"
*   **Segmentierung**: Tab "Segmentierung" → "Segmentieren"

## 📖 Verwendung

### Klassifikation

```python
# Führt ResNet18-basierte Klassifikation durch
# Ausgabe: Wahrscheinlichkeiten für [Riffbarsch, Taucher, Anderer]
# Visualisierung: Balkendiagramm mit Konfidenzwerten
```

### Objekterkennung

```python
# Verwendet YOLOv8 für Objektlokalisierung
# Ausgabe: Bounding Boxes mit Klassenlabels
# Visualisierung: Objektzählung nach Kategorien
```

### Segmentierung

```python
# Adaptive Segmentierung basierend auf Bildcharakteristiken
# Ausgabe: Segmentierungsmaske und Flächenprozentsatz
# Visualisierung: Maske + prozentuale Verteilung
```

## 📁 Projektstruktur

```text
010_Riffbarsch/
├── 📄 README.md                    # Diese Datei
├── 📄 requirements.txt             # Python Dependencies
├── 📄 LICENSE                      # MIT Lizenz
├── 📁 src/                         # Hauptquellcode
│   └── 📁 riffbarsch/              # Python Package
│       ├── 📄 __init__.py          # Package Initialisierung
│       ├── 📄 main.py              # Hauptanwendung (GUI)
│       ├── 📄 main_new.py          # Alternative GUI Version
│       ├── 📄 main_new_fixed.py    # Korrigierte GUI Version
│       ├── 📄 app.py               # Modulare App-Version
│       ├── 📄 config.py            # Konfigurationseinstellungen
│       ├── � mod_000_start.py     # Projektstart-Modul
│       ├── 📄 mod_010_manual_annotation.py          # Manuelle Annotation
│       ├── 📄 mod_020_hard_negative_mining.py       # Hard Negative Mining
│       ├── 📄 mod_030_data_augmentation.py          # Data Augmentation
│       ├── 📄 mod_040_resnet_daten_aufteilen.py     # ResNet Datenaufteilung
│       ├── 📄 mod_050_resnet18_train.py             # ResNet18 Training
│       ├── 📄 mod_060_yolo_daten_aufteilen_v2.py    # YOLO Datenaufteilung V2
│       ├── 📄 mod_070_yolov8n_train.py              # YOLOv8n Training
│       ├── 📄 mod_080_mask_r_cnn_daten_aufteilen.py # Mask R-CNN Daten
│       ├── � mod_090_yolo_sam_maske_erzeugen_turbo.py # YOLO+SAM Masken
│       ├── 📄 mod_100_mask_r_cnn_train.py           # Mask R-CNN Training
│       ├── 📄 mod_101_mask_r_cnn_train_ram_turbo.py # RAM-Turbo Training
│       ├── 📄 mod_110_detectron2_training.py        # Detectron2 Training
│       ├── 📄 mod_970_a_inferenz_vollbild_m_tiling.py # Vollbild-Inferenz
│       ├── 📄 mod_980_inferenz_model_gpu_test.py    # GPU Model Test
│       ├── 📄 mod_990_modell_test_tool.py           # Modell-Test-Tool
│       └── 📄 mod_999_projekt_analyse_v2.py         # Projekt-Analyse
├── 📁 models/                      # Trainierte AI-Modelle
│   ├── 📁 resnet/                  # ResNet18 Klassifikation
│   │   ├── 📄 fisch_v1_lz30_20250923_1451_resnet.pt # ResNet v1 LZ30
│   │   ├── � fisch_v1_lz30_20250923_1532_resnet.pt # ResNet v1 LZ30
│   │   ├── � fisch_v1_lz30_20250923_1618_resnet.pt # ResNet v1 LZ30
│   │   ├── � fisch_v1_lz30_20250923_1716_resnet.pt # ResNet v1 LZ30
│   │   ├── 📄 fisch_v1_lz60_20250923_1810_resnet.pt # ResNet v1 LZ60
│   │   ├── � fisch_v2_Z30_20250924_0727_resnet.pt  # ResNet v2 Z30 (aktuell)
│   │   └── 📄 fisch_v2_Z90_20250923_1939_resnet.pt  # ResNet v2 Z90
│   ├── 📁 yolov8n/                 # YOLO Objekterkennung
│   │   ├── 📁 riffbarsch_taucher_run/               # YOLO Training Run
│   │   │   ├── � args.yaml        # Training Argumente
│   │   │   ├── � confusion_matrix_professional.png # Confusion Matrix
│   │   │   ├── 📄 misclassified_images_analysis.png # Fehlklassifikationen
│   │   │   ├── � training_curves_professional.png  # Training Kurven
│   │   │   └── 📁 weights/         # Trainierte Gewichte
│   │   │       ├── 📄 best.pt      # Bestes Modell
│   │   │       └── 📄 last.pt      # Letztes Modell
│   │   ├── 📁 test_visualisierungen/ # Test Visualisierungen
│   │   └── � yolov8n-cls.pt       # YOLOv8n Basis-Klassifikator
│   ├── 📁 sam_vit/                 # SAM Segmentierung
│   │   ├── � sam_vit_b.pth        # SAM ViT-B Basis
│   │   └── � sam_vit_b_01ec64.pth # SAM ViT-B Vollmodell
│   └── 📁 maskrcnn/                # Mask R-CNN Modelle
│       ├── � mask_rcnn_ram_turbo_epoch_10.pth # Epoch 10
│       ├── � mask_rcnn_ram_turbo_epoch_2.pth  # Epoch 2
│       ├── � mask_rcnn_ram_turbo_epoch_4.pth  # Epoch 4
│       ├── � mask_rcnn_ram_turbo_epoch_6.pth  # Epoch 6
│       ├── � mask_rcnn_ram_turbo_epoch_8.pth  # Epoch 8
│       └── 📄 mask_rcnn_ram_turbo_final.pth    # Finales Modell
├── 📁 datasets/                    # Trainingsdaten (nur Ordnerstruktur)
│   ├── 📁 raw/                     # Rohdaten
│   │   ├── 📁 DNG/                 # RAW-Format Bilder
│   │   └── 📁 JPG/                 # JPEG-Format Bilder
│   ├── 📁 resnet/                  # ResNet Training Split
│   │   ├── 📁 train/               # Training Daten
│   │   │   ├── 📁 hard_negatives/  # Hard Negative Samples
│   │   │   ├── 📁 riffbarsch/      # Riffbarsch Klasse
│   │   │   └── 📁 taucher/         # Taucher Klasse
│   │   ├── 📁 val/                 # Validation Daten
│   │   └── 📁 test/                # Test Daten
│   ├── 📁 yolo_split/              # YOLO Format Daten
│   │   ├── 📁 train/               # Training Split
│   │   │   ├── 📁 images/          # Training Bilder
│   │   │   └── 📁 labels/          # Training Labels (.txt)
│   │   ├── 📁 val/                 # Validation Split
│   │   │   ├── 📁 images/          # Validation Bilder
│   │   │   └── 📁 labels/          # Validation Labels (.txt)
│   │   ├── 📁 test/                # Test Split
│   │   │   ├── 📁 images/          # Test Bilder
│   │   │   └── 📁 labels/          # Test Labels (.txt)
│   │   └── � yolo_riff.yaml       # YOLO Konfiguration
│   ├── 📁 yolo_classification/     # YOLO Klassifikations-Format
│   ├── 📁 maskrcnn_data/           # Mask R-CNN Format
│   │   ├── 📁 train/               # Training Daten
│   │   ├── 📁 val/                 # Validation Daten
│   │   ├── 📁 test/                # Test Daten
│   │   └── 📁 masks/               # Segmentierungsmasken
│   ├── 📁 processed/               # Verarbeitete Datasets
│   │   ├── 📁 hard_negativ/        # Hard Negative Mining Samples (75 Kategorien)
│   │   │   ├── 📁 algae_striped_pattern_underwater/         # Algen mit Streifen
│   │   │   ├── 📁 alien_insect_stripes/                     # Alien Insekten-Streifen
│   │   │   ├── 📁 anemone_tentacles_striped_pattern/        # Anemonen-Tentakel
│   │   │   ├── 📁 aquarium_background_plants_plastic/       # Aquarium Plastikpflanzen
│   │   │   ├── 📁 aquarium_ornament_striped_fish/           # Aquarium Deko-Fische
│   │   │   ├── 📁 artificial_reef_fish_decor/               # Künstliche Riff-Dekoration
│   │   │   ├── 📁 badger_stripes_face/                      # Dachs-Gesichtsstreifen
│   │   │   ├── 📁 banded_sea_snake_underwater/              # Gebänderte Seeschlange
│   │   │   ├── 📁 bee_striped_abdomen_macro/                # Bienen-Abdomen Makro
│   │   │   ├── 📁 black_and_white_butterflyfish_underwater/ # Schwarz-weiße Falterfische
│   │   │   ├── 📁 black_white_striped_alien_humanoid/       # Alien Humanoide
│   │   │   ├── 📁 chipmunk_stripes/                         # Streifenhörnchen
│   │   │   ├── 📁 color_shifted_underwater_coral/           # Farbverschobene Korallen
│   │   │   ├── 📁 coral_reef_textures_underwater/           # Korallenriff-Texturen
│   │   │   ├── 📁 cropped_fish_underwater_photo/            # Angeschnittene Fischfotos
│   │   │   ├── 📁 cuckoo_bird_striped_wings/                # Kuckuck-Flügelstreifen
│   │   │   ├── 📁 dragonfly_striped_wings/                  # Libellen-Flügelstreifen
│   │   │   ├── 📁 fake_fish_toy_underwater/                 # Fake-Fisch Spielzeug
│   │   │   ├── 📁 fantasy_striped_monster/                  # Fantasy-Streifenmonster
│   │   │   ├── 📁 finch_striped_chest/                      # Fink mit gestreifter Brust
│   │   │   ├── 📁 fish_dorsal_fin_closeup_underwater/       # Fisch Rückenflosse Nahaufnahme
│   │   │   ├── 📁 fish_tail_closeup_underwater/             # Fisch Schwanzflosse Nahaufnahme
│   │   │   ├── 📁 gnat_swarm_macro/                         # Mückenschwarm Makro
│   │   │   ├── 📁 greenish_underwater_reef_scene/           # Grünliche Unterwasser-Riffszene
│   │   │   ├── 📁 hawk_striped_tail_feathers/               # Falke mit gestreiften Schwanzfedern
│   │   │   ├── 📁 Heniochus_butterflyfish_reef/             # Heniochus Falterfisch
│   │   │   ├── 📁 hornet_black_yellow_stripes/              # Hornisse schwarz-gelb
│   │   │   ├── 📁 juvenile_golden_trevally_fish/            # Juvenile Goldmakrele
│   │   │   ├── 📁 juvenile_trevally_yellow_stripes/         # Juvenile Makrele gelb gestreift
│   │   │   ├── 📁 lattice_fence_pattern/                    # Gitterzaun-Muster
│   │   │   ├── 📁 mosquito_closeup_striped_legs/            # Mücke Nahaufnahme Beine
│   │   │   ├── 📁 owl_striped_plumage/                      # Eule gestreiftes Gefieder
│   │   │   ├── 📁 partial_fish_body_underwater/             # Partieller Fischkörper
│   │   │   ├── 📁 picket_fence_black_white/                 # Lattenzaun schwarz-weiß
│   │   │   ├── 📁 plastic_aquarium_fish_decoration/         # Plastik Aquarium-Fisch-Deko
│   │   │   ├── 📁 reef_fish_closeup_head/                   # Rifffisch Kopf Nahaufnahme
│   │   │   ├── 📁 reef_rock_banded_pattern/                 # Riffsgestein gebändert
│   │   │   ├── 📁 reef_underwater_shadow_contrast/          # Riff Schatten-Kontrast
│   │   │   ├── 📁 ribboned_sweetlips_fish/                  # Gestreifter Süßlippenfisch
│   │   │   ├── 📁 sci_fi_striped_extraterrestrial/          # Sci-Fi Außerirdische
│   │   │   ├── 📁 scuba_diver_yellow_fins/                  # Taucher gelbe Flossen
│   │   │   ├── 📁 sea_sponge_striped_texture/               # Meeresschwamm gestreift
│   │   │   ├── 📁 shoal_of_striped_tropical_fish/           # Schwarm tropischer Streifenfische
│   │   │   ├── 📁 skunk_black_white_stripes/                # Stinktier schwarz-weiß
│   │   │   ├── 📁 snorkeler_yellow_mask/                    # Schnorchler gelbe Maske
│   │   │   ├── 📁 striped_alien_creature_concept_art/       # Alien Konzeptkunst
│   │   │   ├── 📁 striped_beach_towel/                      # Gestreiftes Strandtuch
│   │   │   ├── 📁 striped_bird_feathers/                    # Gestreifte Vogelfedern
│   │   │   ├── 📁 striped_cat_fur/                          # Gestreiftes Katzenfell
│   │   │   ├── 📁 striped_coral_reef_pattern/               # Gestreiftes Korallenriff-Muster
│   │   │   ├── 📁 striped_fabric_pattern/                   # Gestreiftes Stoffmuster
│   │   │   ├── 📁 striped_fence_wood/                       # Gestreifter Holzzaun
│   │   │   ├── 📁 striped_fly_macro/                        # Gestreifte Fliege Makro
│   │   │   ├── 📁 striped_gate_metal/                       # Gestreiftes Metalltor
│   │   │   ├── 📁 striped_hoverfly_insect/                  # Gestreifte Schwebfliege
│   │   │   ├── 📁 striped_hyena_fur_pattern/                # Streifenhyäne Fellmuster
│   │   │   ├── 📁 striped_okapi_animal/                     # Gestreiftes Okapi
│   │   │   ├── 📁 striped_sea_cucumber/                     # Gestreifte Seegurke
│   │   │   ├── 📁 striped_sea_urchin_spines/                # Gestreifte Seeigel-Stacheln
│   │   │   ├── 📁 striped_towel_beach/                      # Gestreiftes Handtuch
│   │   │   ├── 📁 striped_wall_pattern/                     # Gestreiftes Wandmuster
│   │   │   ├── 📁 sunshade_striped_pattern/                 # Sonnenschirm gestreift
│   │   │   ├── 📁 sweetlips_fish_swarm_underwater/          # Süßlippenfisch-Schwarm
│   │   │   ├── 📁 sweetlips_striped_fish_underwater/        # Gestreifte Süßlippenfische
│   │   │   ├── 📁 tiger_mosquito_stripes/                   # Tigermücke Streifen
│   │   │   ├── 📁 tiger_stripes_closeup/                    # Tiger Streifen Nahaufnahme
│   │   │   ├── 📁 warning_sign_yellow_black_stripes/        # Warnschild gelb-schwarz
│   │   │   ├── 📁 wasp_closeup_stripes/                     # Wespe Nahaufnahme Streifen
│   │   │   ├── 📁 wooden_fence_with_stripes/                # Holzzaun mit Streifen
│   │   │   ├── 📁 yellow_buoy_floating_water/               # Gelbe Boje schwimmend
│   │   │   ├── 📁 yellow_jacket_wasp_flying/                # Gelbwespe fliegend
│   │   │   ├── 📁 yellow_striped_juvenile_reef_fish/        # Gelb gestreifte juvenile Rifffische
│   │   │   ├── 📁 young_golden_trevally_underwater/         # Junge Goldmakrele underwater
│   │   │   ├── 📁 zebra_herd_savanna/                       # Zebraherde Savanne
│   │   │   ├── 📁 zebra_moray_eel_underwater/               # Zebra-Muräne underwater
│   │   │   ├── 📁 zebra_skin_pattern/                       # Zebra Hautmuster
│   │   │   └── 📁 zebra_stripes_closeup/                    # Zebra Streifen Nahaufnahme


│   │   ├── 📁 riffbarsch/              # Positive Samples Riffbarsch (~380 Bilder)
│   │   │   ├── 📄 000001.jpg - 000050.jpg                   # Nummerierte Basis-Bilder
│   │   │   ├── 📄 2025-08-04-*.jpg                          # Zeitstempel-Bilder August 4
│   │   │   ├── 📄 Fisch_#####_*.jpg                         # Anonymisierte Fisch-Bilder
│   │   │   ├── 📄 Fisch_20250808_*.jpg                      # Zeitstempel August 8
│   │   │   ├── 📄 Fisch_20250810_*.jpg                      # Zeitstempel August 10
│   │   │   ├── 📄 Fisch_20250812_*.jpg                      # Zeitstempel August 12
│   │   │   └── 📄 Fisch_20250813_*.jpg                      # Zeitstempel August 13
│   │   ├── 📁 taucher/                 # Positive Samples Taucher (~110 Bilder)
│   │   │   ├── 📄 000001.jpg - 000050.jpg                   # Nummerierte Basis-Bilder
│   │   │   ├── 📄 000r*.jpg, 00r*.jpg, 0r*.jpg              # Rotierte/Varianten
│   │   │   ├── 📄 2025-08-04-*.jpg                          # Zeitstempel-Bilder
│   │   │   ├── 📄 Fisch_#####_*.jpg                         # Anonymisierte Bilder
│   │   │   ├── 📄 Fisch_20250808_*.jpg                      # Zeitstempel-Bilder
│   │   │   └── 📄 r.jpg, rd.jpg                             # Einzelne Varianten




│   │   ├── 📁 riffbarsch_extra/        # Data Augmentation Riffbarsch (24 Augmentations)
│   │   │   ├── 📁 blur/                # Blur-Effekt
│   │   │   ├── 📁 clahe/               # Kontrast-adaptive Histogramm-Equalisierung
│   │   │   ├── 📁 crop/                # Zuschnitte
│   │   │   ├── 📁 distortion/          # Verzerrungen
│   │   │   ├── 📁 elastic/             # Elastische Transformationen
│   │   │   ├── 📁 farbe/               # Farbvariationen
│   │   │   ├── 📁 gelöchert/           # Coarse Dropout (Löcher)
│   │   │   ├── 📁 gespiegelt/          # Horizontale Spiegelung
│   │   │   ├── 📁 gespiegelt_vertikal/ # Vertikale Spiegelung
│   │   │   ├── 📁 grid_distortion/     # Grid-basierte Verzerrung
│   │   │   ├── 📁 hell_dunkel/         # Helligkeitsanpassungen
│   │   │   ├── 📁 perspektive/         # Perspektive-Transformationen
│   │   │   ├── 📁 posterize/           # Posterize-Effekt
│   │   │   ├── 📁 rauschen/            # Rauschen
│   │   │   ├── 📁 rotiert_10grad/      # Rotation 10°
│   │   │   ├── 📁 rotiert_20grad/      # Rotation 20°
│   │   │   ├── 📁 rotiert_30grad/      # Rotation 30°
│   │   │   ├── 📁 rotiert_40grad/      # Rotation 40°
│   │   │   ├── 📁 rotiert_50grad/      # Rotation 50°
│   │   │   ├── 📁 rotiert_60grad/      # Rotation 60°
│   │   │   ├── 📁 rotiert_70grad/      # Rotation 70°
│   │   │   ├── 📁 rotiert_80grad/      # Rotation 80°
│   │   │   ├── 📁 shift_scale/         # Verschiebung und Skalierung
│   │   │   └── 📁 superpixel/          # Superpixel-Transformationen
│   │   ├── 📁 taucher_extra/           # Data Augmentation Taucher (24 Augmentations)
│   │   │   ├── 📁 blur/                # Blur-Effekt
│   │   │   ├── 📁 clahe/               # Kontrast-adaptive Histogramm-Equalisierung
│   │   │   ├── 📁 crop/                # Zuschnitte
│   │   │   ├── 📁 distortion/          # Verzerrungen
│   │   │   ├── 📁 elastic/             # Elastische Transformationen
│   │   │   ├── 📁 farbe/               # Farbvariationen
│   │   │   ├── 📁 gelöchert/           # Coarse Dropout (Löcher)
│   │   │   ├── 📁 gespiegelt/          # Horizontale Spiegelung
│   │   │   ├── 📁 gespiegelt_vertikal/ # Vertikale Spiegelung
│   │   │   ├── 📁 grid_distortion/     # Grid-basierte Verzerrung
│   │   │   ├── 📁 hell_dunkel/         # Helligkeitsanpassungen
│   │   │   ├── 📁 perspektive/         # Perspektive-Transformationen
│   │   │   ├── 📁 posterize/           # Posterize-Effekt
│   │   │   ├── 📁 rauschen/            # Rauschen
│   │   │   ├── 📁 rotiert_10grad/      # Rotation 10°
│   │   │   ├── 📁 rotiert_20grad/      # Rotation 20°
│   │   │   ├── 📁 rotiert_30grad/      # Rotation 30°
│   │   │   ├── 📁 rotiert_40grad/      # Rotation 40°
│   │   │   ├── 📁 rotiert_50grad/      # Rotation 50°
│   │   │   ├── 📁 rotiert_60grad/      # Rotation 60°
│   │   │   ├── 📁 rotiert_70grad/      # Rotation 70°
│   │   │   ├── 📁 rotiert_80grad/      # Rotation 80°
│   │   │   ├── 📁 shift_scale/         # Verschiebung und Skalierung
│   │   │   └── 📁 superpixel/          # Superpixel-Transformationen
│   │   └── 📁 riffbarsch_i_net_data_augmentation/ # Internet Data Augmentation (24 Augmentations)
│   │       ├── 📁 blur/                # Blur-Effekt
│   │       ├── 📁 clahe/               # Kontrast-adaptive Histogramm-Equalisierung
│   │       ├── 📁 crop/                # Zuschnitte
│   │       ├── 📁 distortion/          # Verzerrungen
│   │       ├── 📁 elastic/             # Elastische Transformationen
│   │       ├── 📁 farbe/               # Farbvariationen
│   │       ├── 📁 gelöchert/           # Coarse Dropout (Löcher)
│   │       ├── 📁 gespiegelt/          # Horizontale Spiegelung
│   │       ├── 📁 gespiegelt_vertikal/ # Vertikale Spiegelung
│   │       ├── 📁 grid_distortion/     # Grid-basierte Verzerrung
│   │       ├── 📁 hell_dunkel/         # Helligkeitsanpassungen
│   │       ├── 📁 perspektive/         # Perspektive-Transformationen
│   │       ├── 📁 posterize/           # Posterize-Effekt
│   │       ├── 📁 rauschen/            # Rauschen
│   │       ├── 📁 rotiert_10grad/      # Rotation 10°
│   │       ├── 📁 rotiert_20grad/      # Rotation 20°
│   │       ├── 📁 rotiert_30grad/      # Rotation 30°
│   │       ├── 📁 rotiert_40grad/      # Rotation 40°
│   │       ├── 📁 rotiert_50grad/      # Rotation 50°
│   │       ├── 📁 rotiert_60grad/      # Rotation 60°
│   │       ├── 📁 rotiert_70grad/      # Rotation 70°
│   │       ├── 📁 rotiert_80grad/      # Rotation 80°
│   │       ├── 📁 shift_scale/         # Verschiebung und Skalierung
│   │       └── 📁 superpixel/          # Superpixel-Transformationen
│   └── 📁 downloads/               # Download-Cache für Hard Negative Mining
│       ├── 📁 algae_striped_pattern_underwater/         # Algen mit Streifen
│       ├── 📁 alien_insect_stripes/                     # Alien Insekten-Streifen
│       ├── 📁 anemone_tentacles_striped_pattern/        # Anemonen-Tentakel
│       ├── 📁 aquarium_background_plants_plastic/       # Aquarium Plastikpflanzen
│       ├── 📁 aquarium_ornament_striped_fish/           # Aquarium Deko-Fische
│       ├── 📁 artificial_reef_fish_decor/               # Künstliche Riff-Dekoration
│       ├── 📁 badger_stripes_face/                      # Dachs-Gesichtsstreifen
│       ├── 📁 banded_sea_snake_underwater/              # Gebänderte Seeschlange
│       ├── 📁 bee_striped_abdomen_macro/                # Bienen-Abdomen Makro
│       ├── 📁 black_and_white_butterflyfish_underwater/ # Schwarz-weiße Falterfische
│       ├── 📁 black_white_striped_alien_humanoid/       # Alien Humanoide
│       ├── 📁 chipmunk_stripes/                         # Streifenhörnchen
│       ├── 📁 color_shifted_underwater_coral/           # Farbverschobene Korallen
│       ├── 📁 coral_reef_textures_underwater/           # Korallenriff-Texturen
│       ├── 📁 cropped_fish_underwater_photo/            # Angeschnittene Fischfotos
│       ├── 📁 cuckoo_bird_striped_wings/                # Kuckuck-Flügelstreifen
│       ├── 📁 diver_striped_wetsuit_underwater/         # Taucher mit gestreiftem Anzug
│       ├── 📁 dragonfly_striped_wings/                  # Libellen-Flügelstreifen
│       ├── 📁 fake_fish_toy_underwater/                 # Fake-Fisch Spielzeug
│       ├── 📁 fantasy_striped_monster/                  # Fantasy-Streifenmonster
│       ├── 📁 finch_striped_chest/                      # Fink mit gestreifter Brust
│       ├── 📁 fish_dorsal_fin_closeup_underwater/       # Fisch Rückenflosse Nahaufnahme
│       ├── 📁 fish_tail_closeup_underwater/             # Fisch Schwanzflosse Nahaufnahme
│       ├── 📁 gnat_swarm_macro/                         # Mückenschwarm Makro
│       ├── 📁 greenish_underwater_reef_scene/           # Grünliche Unterwasser-Riffszene
│       ├── 📁 hawk_striped_tail_feathers/               # Falke mit gestreiften Schwanzfedern
│       ├── 📁 Heniochus_butterflyfish_reef/             # Heniochus Falterfisch
│       ├── 📁 hornet_black_yellow_stripes/              # Hornisse schwarz-gelb
│       ├── 📁 juvenile_golden_trevally_fish/            # Juvenile Goldmakrele
│       ├── 📁 juvenile_trevally_yellow_stripes/         # Juvenile Makrele gelb gestreift
│       ├── 📁 lattice_fence_pattern/                    # Gitterzaun-Muster
│       ├── 📁 mosquito_closeup_striped_legs/            # Mücke Nahaufnahme Beine
│       ├── 📁 owl_striped_plumage/                      # Eule gestreiftes Gefieder
│       ├── 📁 partial_fish_body_underwater/             # Partieller Fischkörper
│       ├── 📁 picket_fence_black_white/                 # Lattenzaun schwarz-weiß
│       ├── 📁 plastic_aquarium_fish_decoration/         # Plastik Aquarium-Fisch-Deko
│       ├── 📁 reef_fish_closeup_head/                   # Rifffisch Kopf Nahaufnahme
│       ├── 📁 reef_rock_banded_pattern/                 # Riffsgestein gebändert
│       ├── 📁 reef_underwater_shadow_contrast/          # Riff Schatten-Kontrast
│       ├── 📁 ribboned_sweetlips_fish/                  # Gestreifter Süßlippenfisch
│       ├── 📁 sci_fi_striped_extraterrestrial/          # Sci-Fi Außerirdische
│       ├── 📁 scuba_diver_yellow_fins/                  # Taucher gelbe Flossen
│       ├── 📁 sea_sponge_striped_texture/               # Meeresschwamm gestreift
│       ├── 📁 sergeant_major_fish_school/               # Sergeant Major Fischschwarm
│       ├── 📁 shoal_of_striped_tropical_fish/           # Schwarm tropischer Streifenfische
│       ├── 📁 skunk_black_white_stripes/                # Stinktier schwarz-weiß
│       ├── 📁 snorkeler_yellow_mask/                    # Schnorchler gelbe Maske
│       ├── 📁 striped_alien_creature_concept_art/       # Alien Konzeptkunst
│       ├── 📁 striped_beach_towel/                      # Gestreiftes Strandtuch
│       ├── 📁 striped_bird_feathers/                    # Gestreifte Vogelfedern
│       ├── 📁 striped_cat_fur/                          # Gestreiftes Katzenfell
│       ├── 📁 striped_coral_reef_pattern/               # Gestreiftes Korallenriff-Muster
│       ├── 📁 striped_fabric_pattern/                   # Gestreiftes Stoffmuster
│       ├── 📁 striped_fence_wood/                       # Gestreifter Holzzaun
│       ├── 📁 striped_fly_macro/                        # Gestreifte Fliege Makro
│       ├── 📁 striped_gate_metal/                       # Gestreiftes Metalltor
│       ├── 📁 striped_hoverfly_insect/                  # Gestreifte Schwebfliege
│       ├── 📁 striped_hyena_fur_pattern/                # Streifenhyäne Fellmuster
│       ├── 📁 striped_okapi_animal/                     # Gestreiftes Okapi
│       ├── 📁 striped_sea_cucumber/                     # Gestreifte Seegurke
│       ├── 📁 striped_sea_urchin_spines/                # Gestreifte Seeigel-Stacheln
│       ├── 📁 striped_towel_beach/                      # Gestreiftes Handtuch
│       ├── 📁 striped_wall_pattern/                     # Gestreiftes Wandmuster
│       ├── 📁 sunshade_striped_pattern/                 # Sonnenschirm gestreift
│       ├── 📁 sweetlips_fish_swarm_underwater/          # Süßlippenfisch-Schwarm
│       ├── 📁 sweetlips_striped_fish_underwater/        # Gestreifte Süßlippenfische
│       ├── 📁 tiger_mosquito_stripes/                   # Tigermücke Streifen
│       ├── 📁 tiger_stripes_closeup/                    # Tiger Streifen Nahaufnahme
│       ├── 📁 underwater_backscatter_photography/       # Unterwasser Rückstreuung
│       ├── 📁 underwater_low_light_photography/         # Unterwasser Low-Light
│       ├── 📁 warning_sign_yellow_black_stripes/        # Warnschild gelb-schwarz
│       ├── 📁 wasp_closeup_stripes/                     # Wespe Nahaufnahme Streifen
│       ├── 📁 wooden_fence_with_stripes/                # Holzzaun mit Streifen
│       ├── 📁 yellow_buoy_floating_water/               # Gelbe Boje schwimmend
│       ├── 📁 yellow_jacket_wasp_flying/                # Gelbwespe fliegend
│       ├── 📁 yellow_striped_juvenile_reef_fish/        # Gelb gestreifte juvenile Rifffische
│       ├── 📁 young_golden_trevally_underwater/         # Junge Goldmakrele underwater
│       ├── 📁 zebra_herd_savanna/                       # Zebraherde Savanne
│       ├── 📁 zebra_moray_eel_underwater/               # Zebra-Muräne underwater
│       ├── 📁 zebra_skin_pattern/                       # Zebra Hautmuster
│       └── 📁 zebra_stripes_closeup/                    # Zebra Streifen Nahaufnahme
├── 📁 img/                         # Dokumentations-Bilder
│   ├── 📄 Figure_1.png             # Interaktive Klassifikation
│   ├── 📄 Figure_2.png             # Workflow Pipeline
│   ├── 📄 Figure_3.png             # Data Augmentation
│   ├── 📄 Figure_4.png             # YOLO Training
│   ├── 📄 Figure_5.png             # Segmentierung
│   └── 📄 banner_frank_albrecht.jpg # Header Banner
├── 📁 docs/                        # HTML Dokumentation
│   ├── 📄 index.html               # Hauptnavigation
│   ├── 📄 main.html                # Projektübersicht
│   ├── 📄 mod_000_start.html       # Startseite Dokumentation
│   ├── 📄 mod_010_manual_annotation.html    # Manuelle Annotation Docs
│   ├── 📄 mod_020_hard_negative_mining.html # Hard Negative Mining Docs
│   ├── 📄 mod_030_data_augmentation.html    # Data Augmentation Docs
│   ├── 📄 mod_040_resnet_daten_aufteilen.html # ResNet Daten Docs
│   ├── 📄 mod_050_resnet18_train.html       # ResNet Training Docs
│   ├── 📄 mod_060_yolo_daten_aufteilen_v2.html # YOLO Daten V2 Docs
│   ├── 📄 mod_070_yolov8n_train.html        # YOLOv8n Training Docs
│   ├── 📄 mod_080_mask_r_cnn_daten_aufteilen.html # Mask R-CNN Daten Docs
│   ├── 📄 mod_090_yolo_sam_maske_erzeugen_turbo.html # YOLO+SAM Docs
│   ├── 📄 mod_100_mask_r_cnn_train.html     # Mask R-CNN Training Docs
│   ├── 📄 mod_101_mask_r_cnn_train_ram_turbo.html # RAM-Turbo Docs
│   └── 📄 mod_110_detectron2_training.html  # Detectron2 Training Docs
├── 📁 notebooks/                   # Jupyter Notebooks (geplant)
│   └── 📄 .gitkeep                 # Placeholder für Git
└── 📁 scripts/                     # Zusätzliche Skripte (geplant)
    └── � .gitkeep                 # Placeholder für Git
```

## 🤖 Modelle

### ResNet18 Klassifikator

*   **Architektur**: ResNet18 mit angepasstem FC-Layer
*   **Klassen**: 3 (Riffbarsch, Taucher, Anderer)
*   **Input**: 224×224 RGB Bilder
*   **Vortraining**: ImageNet
*   **Genauigkeit**: ~85% auf Testdaten

### YOLOv8n Objektdetektor

*   **Architektur**: YOLOv8 Nano
*   **Klassen**: Riffbarsch, Taucher, Anderer
*   **Input**: Variable Auflösung (empfohlen 640×640)
*   **mAP**: ~0.72 @ IoU 0.5

### Segmentierungsalgorithmus

*   **Adaptive Strategie**: Basiert auf Bildhelligkeits- und Kontrastanalyse
*   **Modi**: Kontur-basiert, Kantenerkennung, Elliptisch
*   **Morphologie**: Opening/Closing für Glättung

## ⚙️ Konfiguration

### Modellpfade anpassen

```python
# In src/010_Riffbarsch/config.py
RESNET_PATH = "path/to/your/resnet/model.pt"
YOLO_PATH = "path/to/your/yolo/model.pt"
```

### GUI-Einstellungen

```python
# Fenstergröße
WINDOW_SIZE = "1400x800"

# Farben
COLORS = {
    'primary': '#2c3e50',
    'success': '#27ae60',
    # ...
}
```

### Analyseeinstellungen

```python
# YOLO Konfidenz-Schwelle
YOLO_CONFIDENCE_THRESHOLD = 0.25

# Segmentierungsparameter
SEGMENT_ELLIPSE_SCALE = (0.35, 0.25)
```

## 🛠️ Entwicklung

### Entwicklungsumgebung einrichten

```bash
# Development Dependencies installieren
pip install -r requirements-dev.txt

# Pre-commit hooks installieren
pre-commit install
```

### Tests ausführen

```bash
# Alle Tests
pytest

# Mit Coverage
pytest --cov=src/010_Riffbarsch

# Spezifische Tests
pytest tests/test_01_unit.py
```

### Code-Qualität

```bash
# Linting
flake8 src/
black src/
isort src/

# Type checking
mypy src/
```

### Neues Modell hinzufügen

1.  Modell in `models/` Ordner speichern
2.  Pfad in `config.py` aktualisieren
3.  Loader-Funktion erstellen
4.  GUI-Integration in entsprechendem Tab

## 🔧 Troubleshooting

### Häufige Probleme

#### `ModuleNotFoundError`

```bash
# Lösung: PYTHONPATH setzen
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# oder
python -m src.010_Riffbarsch.main
```

#### `CUDA out of memory`

```bash
# Lösung: Batch Size reduzieren oder CPU verwenden
# In config.py:
device = torch.device("cpu")  # Statt "cuda"
```

#### Modell nicht gefunden

```bash
# Prüfe Pfade in config.py
# Stelle sicher dass .pt Dateien existieren
ls models/resnet/
ls models/yolov8n/riffbarsch_taucher_run/weights/
```

### Debug-Modus aktivieren

```python
# In main.py:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance-Optimierung

*   **GPU verwenden**: Stelle sicher, dass CUDA installiert ist
*   **Bildgröße reduzieren**: Für schnellere Verarbeitung
*   **Batch Processing**: Für mehrere Bilder gleichzeitig

## 🤝 Beitragen

### Wie du beitragen kannst

1.  Fork das Repository
2.  Branch für deine Änderungen: `git checkout -b feature/AmazingFeature`
3.  Commit deine Änderungen: `git commit -m 'Add AmazingFeature'`
4.  Push zum Branch: `git push origin feature/AmazingFeature`
5.  Pull Request öffnen

### Code-Standards

*   PEP 8 Python Style Guide befolgen
*   Type Hints verwenden wo möglich
*   Docstrings für alle öffentlichen Funktionen
*   Tests für neue Features schreiben
*   Meaningful Commit Messages verwenden

### Issue Guidelines

*   **Bug Reports**: Template verwenden, Steps to reproduce angeben
*   **Feature Requests**: Use Case und erwarteten Nutzen beschreiben
*   **Questions**: Stack Overflow bevorzugen für generelle Fragen

## 📚 Ressourcen

### Wissenschaftliche Referenzen

*   [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
*   [YOLOv8: A Real-Time Object Detection Algorithm (YOLO)](https://docs.ultralytics.com/yolov8/)
*   Marine Life Detection in Underwater Images

### Nützliche Links

*   NN Visualisierung: [https://adamharley.com/nn_vis/cnn/3d.html](https://adamharley.com/nn_vis/cnn/3d.html)
*   KI Explainability: [https://lrpserver.hhi.fraunhofer.de/](https://lrpserver.hhi.fraunhofer.de/)
*   Modell Visualisierung: [https://netron.app/](https://netron.app/)
*   Ultralytics Docs: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
*   PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

### Datensätze

*   FishNet - Large-scale fish dataset
*   Fish4Knowledge - Underwater fish recognition
*   NOAA Fisheries - Government fish data

## 📄 Lizenz

Dieses Projekt ist unter der MIT License lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

```text
MIT License

Copyright (c) 2025 Riffbarsch AI Project

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

```text
YOLOv8-Modelle werden unter AGPL-3.0- und Enterprise-Lizenzen bereitgestellt.
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

```



## 👥 Autoren & Mitwirkende

*   **Hauptentwickler**: @Watchkido
*   **KI-Experte**: Arbeitet an Computer Vision
*   **Data Analyst**: Beratung für maritime Anwendungen

## 📈 Roadmap

### Version 2.0 (Geplant)

*   Web-Interface mit Flask/FastAPI
*   REST API für externe Integration
*   Batch Processing für große Bildermengen
*   Model Ensemble für höhere Genauigkeit
*   Real-time Video Analyse
*   Cloud Deployment (Docker/Kubernetes)

### Version 1.5 (In Entwicklung)

*   Improved Segmentation mit SAM Integration
*   Data Augmentation Pipeline
*   Model Interpretability mit Grad-CAM
*   Performance Metrics Dashboard

<div align="center">
⭐ Wenn dir dieses Projekt gefällt, gib ihm einen Stern! ⭐

![Last Commit](https://img.shields.io/github/last-commit/Watchkido/010_Riffbarsch)
![Contributors](https://img.shields.io/github/contributors/Watchkido/010_Riffbarsch)

</div>

