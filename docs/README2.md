# 🐠 Pipeline zur Detektion von Riffbarschen auf Unterwasserbildern

## 1. 🎯 Zieldefinition
- Automatische Erkennung von Riffbarschen auf hochauflösenden JPG-Bildern
- Fokus auf eine einzelne Zielklasse: „Riffbarsch“
- Ausgabe: Bounding Boxes mit Position und Konfidenz

---

## 2. 🗂️ Datensatz
- 1000 hochauflösende Unterwasserbilder (JPG)
- Manuelle Annotation mit Bounding Boxes (YOLO-Format)
- Tools: LabelImg oder CVAT
- Datenaufteilung: 70% Training, 15% Validierung, 15% Test

---

## 3. 🧼 Vorverarbeitung
- Farbkorrektur: Weißabgleich, Kontrastanpassung
- Tiling: Bilder in 1024×1024-Patches mit 20% Überlappung
- Duplikaterkennung: Perceptual Hashing zur Vermeidung von Leckage

---

## 4. 🧪 Augmentierung
- Geometrisch: Flip, Rotation, Cropping
- Fotometrisch: Farb-Jitter, Helligkeit/Kontrast, Rauschen
- Mosaik/MixUp: Kombinierte Bilder zur Erhöhung der Diversität
- Ziel: künstliche Erweiterung des Datensatzes auf >10.000 Varianten

---

## 5. 🧠 Modellwahl
- YOLOv8 oder Tiny-YOLOv3 (reines PyTorch)
- Backbone: Small oder Medium, je nach GPU
- Ankerfreie Architektur bevorzugt (robust bei kleinen Objekten)

---

## 6. 🏋️‍♂️ Training
- Eingabebildgröße: 1024×1024
- Batch-Größe: abhängig vom VRAM
- Optimierer: AdamW oder SGD
- Lernrate: 1e-3 mit Cosine Decay
- Epochen: 100–300 mit Early Stopping
- Loss-Funktion: Focal Loss zur Behandlung von Klassenungleichgewicht

---

## 7. 📊 Evaluation
- Hauptmetriken:
  - mAP@0.5
  - mAP@0.5:0.95
  - Precision & Recall
- Visualisierung: PR-Kurven, Fehleranalyse (False Positives/Negatives)
- Analyse nach Objektgröße: mAP-small, mAP-medium

---

## 8. 🔍 Inferenz
- Sliding Window über Vollbilder (1024er Kacheln)
- Confidence-Threshold: z. B. 0.25
- Non-Maximum Suppression (NMS) über Kachelgrenzen hinweg
- Ausgabe: Liste von Bounding Boxes mit Konfidenz

---

## 9. 🧩 Postprocessing
- Optionales Tracking: z. B. DeepSORT für Videos
- Hard Negatives: gezielte Beispiele zur Reduktion von Fehlalarmen
- Domain Randomization: Variation von Licht, Farbe, Trübung zur Generalisierung

---

## 10. 🛠️ Tools & Libraries
- Annotation: LabelImg, CVAT
- Bildverarbeitung: Pillow, OpenCV
- Augmentierung: imgaug, Albumentations
- Modelltraining: PyTorch (reines Python)
- Evaluation: eigene mAP-Berechnung oder pycocotools

---

## ✅ Zusammenfassung
- Kompakte, robuste Pipeline für kleine Datensätze
- Fokus auf Unterwasserbedingungen und kleine Zielobjekte
- Modularer Aufbau für einfache Erweiterung und Anpassung

