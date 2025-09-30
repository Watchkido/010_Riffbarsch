# Wie Mask R-CNN arbeitet

Mask R-CNN ist ein neuronales Netzwerk für die **Objekterkennung und Segmentierung**. Es kann nicht nur erkennen, **welche Objekte** in einem Bild sind, sondern auch **welche Pixel genau** zu jedem Objekt gehören.

---

## Datenaufbereitung

### Ausgangsmaterial:
- Originalbilder (z. B. Unterwasseraufnahmen)
- Bounding Boxes von Objekten (z. B. erzeugt durch YOLO)
- Pixelgenaue Masken der Objekte (z. B. erzeugt durch SAM)

---

### Beispiel für Metadaten eines Bildes:
```json
{
  "image": "000018.jpg",
  "instances": [
    {
      "mask_file": "000018_riffbarsch_00.png",
      "class_id": 1,
      "class_name": "riffbarsch",
      "bbox": [0, 0, 975, 1390]
    }
  ]
}
```

**Erläuterung:**
- `"image"`: Name des Originalbilds
- `"instances"`: Liste aller Objekte im Bild
- `"mask_file"`: Pfad zur pixelgenauen Maske (0 = Hintergrund, 255 = Objekt)
- `"class_id"`: Klassenzuordnung für Mask R-CNN (0 = Hintergrund, 1 = Riffbarsch, 2 = Taucher)
- `"class_name"`: Lesbarer Name der Klasse
- `"bbox"`: Bounding Box `[x_min, y_min, x_max, y_max]` in Pixeln

---

## Trainingsprozess

1. **YOLO erzeugt Bounding Boxes**  
   - Liefert grobe Objektpositionen.
2. **SAM erstellt exakte Masken**  
   - Pixelgenaue Objektmasken basierend auf YOLO-Boxen.
3. **Mask R-CNN lernt**:
   - Die genaue Form der Objekte (Segmentierung)
   - Die Position der Objekte (Bounding Box)
   - Die Klasse der Objekte (Label)

---

## Zusammenfassung

- YOLO → liefert **grobe Objektkoordinaten**
- SAM → erstellt **pixelgenaue Masken**
- Mask R-CNN → lernt **Segmentierung, Klassifikation und Lokalisierung gleichzeitig**

So kann Mask R-CNN später auf neuen Bildern automatisch erkennen, **wo die Objekte sind, welche Klasse sie haben und welche Pixel dazu gehören**.

