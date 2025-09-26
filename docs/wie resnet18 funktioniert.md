# Wie ResNet-18 arbeitet

ResNet-18 ist ein tiefes neuronales Netzwerk für die **Bildklassifizierung**. Es wurde entwickelt, um das Problem des **Verschwindenden Gradienten** in sehr tiefen Netzen zu lösen, indem es sogenannte **"Residual Connections"** (Übersprungverbindungen) einführt.

---

## Aufbau von ResNet-18

### Grundidee: Residualblöcke
Statt eine komplexe Funktion \( F(x) \) zu lernen, lernt ResNet die **Abweichung** vom Eingabewert:
\[
F(x) = H(x) - x
\]
Die Ausgabe eines Blocks wird dann:
\[
H(x) = F(x) + x
\]

### Architekturübersicht:
- **18 Schichten** (daher der Name)
- **4 Hauptblöcke** mit je 2 Residualblöcken
- Am Ende: **Global Average Pooling** und **Fully Connected Layer** für Klassifizierung

---

## Warum ResNet-18 wichtig ist

- Ermöglicht das Training sehr tiefer Netzwerke
- Verhindert das Verschwinden des Gradienten
- Gute Balance zwischen Genauigkeit und Rechenaufwand
- Oft als **Backbone** in anderen Architekturen (wie Mask R-CNN) verwendet

---

## Zusammenfassung

ResNet-18 klassifiziert Bilder, indem es:
- **Residualblöcke** zur Vermeidung von Gradientenproblemen nutzt
- Eine Tiefe von **18 Schichten** hat
- Häufig als **Feature-Extractor** in Detektions- und Segmentierungsnetzwerken dient

---

## Beispiel-Anwendung

- Bilderkennung (z. B. ImageNet)
- Feature-Extraction für Objekterkennung
- Transfer Learning für kleine Datensätze