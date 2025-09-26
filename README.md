# 🐟 010_Riffbarsch

## Projektübersicht
Pipeline für Unterwasser-Fischdetektion (Riffbarsch) von Datensatz bis Deployment.


## Webseirten
     https://adamharley.com/nn_vis/cnn/3d.html
   wi efindet das programm die richtigen punkte inm bild. https://lrpserver.hhi.fraunhofer.de/
   Modell anzeigen lassen: https://netron.app/
---

## 1. Datensatz & Labeling
- **Zielklasse:** Riffbarsch (ein Klassenlabel reicht für den Anfang)
- **Annotation:** Bounding Boxes (YOLO-/COCO-Format)
- **Tools:** Label Studio, CVAT, Roboflow
- **Split:** 70% Training, 15% Validierung, 15% Test
- **Qualität:** Mindestens 5–10% doppelt prüfen (Unterwasserbilder sind tricky: Blendlicht, Trübung, Partikel)
Input: Hochauflösende JPGs

Output: Positionen der Riffbarsche als Bounding Boxes

Modell: Ein einfaches CNN + Region Proposal oder ein Tiny-YOLOv3 in reinem PyTorch

Annotation: Du brauchst manuelle Labels (z. B. mit LabelImg oder CVAT) im YOLO- oder Pascal VOC-Format
---

## 2. Vorverarbeitung
- **Farbkorrektur:** Weißabgleich, Gray-World, CLAHE (optional, offline)
- **Auflösung:** Originale behalten, fürs Training in Patches/Tiles schneiden (z. B. 1024×1024, 10–20% Überlappung)
- **Duplikate entfernen:** Near-duplicate detection (z. B. Perceptual Hashing)

---

## 3. Augmentations (bei <1000 Bildern essentiell)
- **Geometrisch:** Horizontal-Flip, Rotation, Random Crop mit IoU-Erhalt
- **Fotometrisch:** Helligkeit/Kontrast, Farb-Jitter, Gaussian Noise, Motion Blur
- **Mosaik/MixUp:** Sparsam einsetzen
- **Kleine-Objekt-Fokus:** Random scale up/down, kleine Fische nicht verlieren

---

## 4. Modellwahl
- **Startpunkt:** YOLOv8/YOLOv5 oder RT-DETR-Small (vortrainiert)
- **Backbone:** Small/Medium, Multi-Scale-FPN/PAFPN für kleine Ziele
- **Ankerfrei vs. Ankerbasiert:** YOLOv8/RT-DETR sind robust, beides testen

---

## 5. Training
- **Bildgröße:** 1024 oder 1280 (bei Tiling: 1024)
- **Batch/GPU:** So groß wie VRAM erlaubt (Gradient Accumulation falls nötig)
- **Optimierer:** SGD oder AdamW, Lernrate 1e-3 (Cosine Decay oft gut)
- **Epochen:** 100–300, Early Stopping nach Val mAP
- **Klassenungleichgewicht:** Focal Loss aktivieren
- **Class-agnostic NMS:** Anfangs ok, später klassenspezifisch

---

## 6. Metriken & Monitoring
- **Primär:** mAP@0.50 und mAP@0.50:0.95
- **Sekundär:** Precision/Recall, PR-Kurve, Fehlerinspektion (FP bei Korallen/Steinen, FN bei verdeckten Fischen)
- **Per-Size-Analyse:** mAP small/medium (wichtig bei mini-Fischen)

---

## 7. Inferenz & Postprocessing
- **Sliding Window/Tiling:** Fensterweise detektieren, NMS über Kachelgrenzen
- **Confidence-/IoU-Threshold:** Sweepen (z. B. conf 0.15–0.35; IoU-NMS 0.5–0.7)
- **Tracking (optional):** DeepSORT/ByteTrack für Videos

---

## 8. Iteratives Verbessern
- **Active Learning:** Unsichere Beispiele nachannotieren
- **Hard Negatives:** Verwechslungsstrukturen gezielt hinzufügen
- **Domain Randomization:** Mehr Variation in Farben/Trübung/Beleuchtung

---

## Hinweise
- Für Fragen und Erweiterungen: Siehe /notebooks und /scripts
- Kontakt: Projektleitung oder GitHub-Issues

## Praxis-Tipps für Unterwasser-Robustheit
- Weißabgleich-Varianz: Trainiere mit leicht unterschiedlichen WB-Versionen derselben Szene als Augmentation, 
  um Farbdrift zu entkoppeln.
- Backscatter/Partikel: Leichte Additive Noise- und Speckle-Simulation verbessert die Robustheit.
- Falsche Positive: Korallenäste, Schatten und Felsen sind typische Täuschungen. 
  Sammle ein “hard negatives”- Subset und füge es dem Training hinzu.
- Kleine Ziele: Prüfe mAP-small. Wenn schwach, erhöhe Eingabeauflösung, nutze stärkere PAFPN-Varianten, 
  oder reduziere Stride der ersten Stufen.

