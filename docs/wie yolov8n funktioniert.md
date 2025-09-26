# Wie YOLOv8n arbeitet

YOLOv8n (You Only Look Once, Version 8 Nano) ist ein **Echtzeit-Objektdetektor**, der besonders **schnell und ressourcenschonend** ist. Es detektiert Objekte in einem Bild in einem einzigen Durchgang durch das Netzwerk.

---

## Funktionsweise von YOLOv8n

### Ein-Stufen-Detektion (One-Stage)
Im Gegensatz zu zwei-stufigen Detektoren (wie Mask R-CNN) führt YOLO:
- **Klassifikation**
- **Bounding-Box-Regression**
- **Objekt-Wahrscheinlichkeit**

**in einem Schritt** durch.

### Architekturmerkmale:
- **Backbone**: CSPDarknet (leicht und effizient)
- **Neck**: PAN-FPN (Feature-Pyramid Network für Multi-Scale-Detektion)
- **Head**: Ankerfreie Detektion (vereinfacht die Box-Vorhersage)

---

## Trainingsprozess

1. **Eingabe**: Bild wird auf feste Größe skaliert (z. B. 640×640)
2. **Grid-Aufteilung**: Bild wird in Zellen unterteilt
3. **Vorhersage**: Jede Zelle sagt Bounding Boxes, Klassenwahrscheinlichkeiten und Objektness-Score vorher
4. **Verlustfunktion**: Kombiniert Box-, Klassen- und Objektness-Fehler

---

## Vorteile von YOLOv8n

- **Sehr schnell** (ideal für Echtzeitanwendungen)
- **Klein und effizient** (nano-Version für eingeschränkte Hardware)
- **Hohe Genauigkeit** trotz geringer Größe
- **Einfache Integration** in Produktionssysteme

---

## Zusammenfassung

YOLOv8n detektiert Objekte in Echtzeit, indem es:
- Das Bild in einem **einzigen Durchgang** verarbeitet
- Eine **ankerfreie Methode** verwendet
- **Multi-Scale-Features** für kleine und große Objekte nutzt
- Ideal ist für **Mobile Geräte und Edge-Geräte**

---

## Beispiel-Anwendung

- Live-Objekterkennung in Videos
- Autonomous Driving
- Überwachungssysteme
- Robotersteuerung