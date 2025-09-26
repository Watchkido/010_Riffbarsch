# YOLOv8 Trainer Parameter Übersicht

Diese Tabelle fasst die wichtigsten Parameter der **Ultralytics YOLOv8 Trainer Engine** zusammen.
Sie dient als Spickzettel für Präsentationen oder Training.

---

## ⚙️ Wichtige Trainer-Parameter (YOLOv8)

| Parameter | Bedeutung |
|-----------|-----------|
| **epochs** | Anzahl Trainingsdurchläufe über den gesamten Datensatz (mehr = länger, oft bessere Genauigkeit). |
| **batch** | Wie viele Bilder pro Schritt gleichzeitig verarbeitet werden (größer = stabiler, braucht mehr RAM). |
| **imgsz** | Bildgröße (z. B. 640x640) – beeinflusst Genauigkeit und Geschwindigkeit. |
| **optimizer** | Algorithmus, wie die Gewichte angepasst werden (SGD, Adam, AdamW). |
| **lr0** | Start-Lernrate – steuert, wie stark Gewichte angepasst werden. |
| **lrf** | Endwert der Lernrate (wird über Training abgesenkt → "learning rate decay"). |
| **momentum** | Trägheit bei SGD/Adam – bestimmt, wie stark vergangene Updates Einfluss haben. |
| **weight_decay** | Regulierung gegen Overfitting (wie stark große Gewichte bestraft werden). |
| **warmup_epochs** | Erste Epochen mit kleinerer Lernrate zum stabilen Start. |
| **patience** | Frühes Stoppen, wenn sich die Validierungsleistung nicht verbessert. |

---

## 🎨 Datenaugmentation (künstliche Bildveränderungen)

| Parameter | Bedeutung |
|-----------|-----------|
| **hsv_h / hsv_s / hsv_v** | Veränderung von Farbton, Sättigung, Helligkeit (macht Modell robuster gegen Lichtverhältnisse). |
| **fliplr / flipud** | Horizontal/Vertikal spiegeln (z. B. Taucher auf der anderen Seite). |
| **degrees** | Zufällige Rotation. |
| **scale** | Bild skalieren (vergrößern/verkleinern). |
| **shear** | Schrägziehen. |
| **translate** | Verschieben im Bild. |
| **mosaic** | Mehrere Bilder zu einem kombiniert → starkes Augmentierungsverfahren für Objekterkennung. |
| **mixup / cutmix** | Bilder oder Bildteile überlagern → erschwert Overfitting. |
| **erasing** | Teile des Bildes löschen → Robustheit gegen verdeckte Objekte. |

---

## 🔧 Weitere Einstellungen

| Parameter | Bedeutung |
|-----------|-----------|
| **device** | CPU oder GPU (z. B. `cuda:0`). |
| **pretrained** | Ob mit vortrainierten Gewichten gestartet wird. |
| **classes** | Welche Klassen trainiert werden sollen (z. B. nur Taucher). |
| **single_cls** | Alles als eine Klasse behandeln. |
| **conf** | Konfidenzschwelle bei Inferenz (z. B. nur Objekte mit >0.25 Wahrscheinlichkeit zeigen). |
| **iou** | Intersection-over-Union-Schwelle beim NMS → trennt sich überlappende Boxen. |
| **nms / agnostic_nms** | Non-Maximum-Suppression (Zusammenfassen überlappender Boxen). |
| **save / save_period** | Modelle & Ergebnisse speichern (z. B. alle 10 Epochen). |
| **plots** | Trainingsplots (Loss, mAP etc.) erstellen. |
| **verbose** | Ausführliche Logs. |

---

👉 **Zusammengefasst**:  
- Oben: wie dein Training lernt (**epochs, batch, optimizer, lr0**)  
- Mitte: wie deine Daten variiert werden (**Augmentation**)  
- Unten: praktische Optionen für Speicher, Logs, Geräte etc.
