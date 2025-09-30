# Migration von Classification zu Detection Model

## Wichtige Änderungen in mod_070_yolov8n_train.py

### 1. Model-Typ geändert
- **Vorher:** `YOLO("yolov8n-cls.pt")` (Classification)
- **Nachher:** `YOLO("yolov8n.pt")` (Detection)

### 2. Dataset-Struktur
**Vorher (Classification):**
```
dataset/
├── train/
│   ├── riffbarsch/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── taucher/
│       ├── img1.jpg
│       └── img2.jpg
```

**Nachher (Detection):**
```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
```

### 3. Annotation Format (neu für Detection)
Jede `.txt` Datei enthält pro Zeile:
```
class_id x_center y_center width height
```
Beispiel (`img1.txt`):
```
0 0.5 0.3 0.2 0.4    # Riffbarsch mittig
1 0.2 0.7 0.15 0.25  # Taucher links unten
```

**Koordinaten sind normalisiert (0.0 bis 1.0):**
- `x_center`: Horizontale Mitte (0=links, 1=rechts)
- `y_center`: Vertikale Mitte (0=oben, 1=unten)  
- `width`: Breite relativ zur Bildbreite
- `height`: Höhe relativ zur Bildhöhe

### 4. YAML-Konfiguration
**Detection YAML:**
```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images
names:
  0: riffbarsch
  1: taucher
nc: 2
```

### 5. Training-Parameter angepasst
- **Mosaic Augmentation**: Für bessere Detection
- **MixUp**: Für Generalisierung
- **Moderatere Rotation**: Bounding Boxes bleiben korrekt

### 6. Metriken geändert
**Vorher (Classification):**
- Accuracy, Precision, Recall pro Klasse
- Confusion Matrix

**Nachher (Detection):**
- **mAP@0.5**: Mean Average Precision bei IoU 0.5
- **mAP@0.5:0.95**: mAP über IoU-Bereiche 0.5-0.95
- **Precision/Recall**: Für Detection-Aufgabe
- Klassenweise Average Precision

## Was Sie noch tun müssen:

### 1. Annotationen erstellen
Sie benötigen Bounding Box Annotationen für Ihre Bilder. Tools:
- **LabelImg**: Grafisches Annotationstool
- **CVAT**: Web-basiertes Annotationstool
- **Roboflow**: Online-Platform mit Auto-Annotation

### 2. Dataset konvertieren
Aktuelle Classification-Bilder zu Detection-Format:
1. Bilder von `train/class_name/` nach `train/images/` kopieren
2. Für jedes Bild eine `.txt` Datei mit Bounding Boxes erstellen

### 3. Training starten
```bash
cd "E:\dev\projekt_python_venv\010_Riffbarsch\src\riffbarsch"
python mod_070_yolov8n_train.py
```

## Ergebnis
Das trainierte Model kann jetzt:
- ✅ **Bounding Boxes** um Objekte zeichnen
- ✅ **Mehrere Objekte** pro Bild erkennen
- ✅ **Objektpositionen** bestimmen
- ✅ In `main_v5.py` **funktionieren** (results[0].boxes wird nicht None sein)

## Unterschied zu vorher
- **Classification**: "Dieses Bild zeigt einen Riffbarsch"
- **Detection**: "In diesem Bild ist ein Riffbarsch bei Koordinaten (100,50,200,150)"