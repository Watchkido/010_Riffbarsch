# 🎤 Präsentation: Manual Annotation (Mod 010)

## Fach- und Fremdworte

| Fremdwort / Fachbegriff        | Erklärung (Deutsch, 1 Satz) |
|--------------------------------|-----------------------------|
| Manual Annotation Pipeline     | Menschlicher Prozess zum händischen Labeln von Bildern für KI-Training |
| Ground Truth Generation        | Erzeugung der "richtigen Antworten", die als Referenz fürs KI-Training dienen |
| Binary Classification          | Ein Klassifikationssystem mit nur zwei Klassen (Ja/Nein) |
| Interactive Processing         | Nutzerinteraktion zur Steuerung des Ablaufs |
| File Organization              | Automatisches Einsortieren von Dateien in passende Ordner |
| Supervised Learning            | Überwachtes Lernen, bei dem die KI mit menschlich gelabelten Daten trainiert wird |
| Domain Expertise               | Fachwissen von Menschen, das ins Training einfließt |
| Training Foundation            | Basisdaten, auf denen das spätere Modell aufbaut |
| Class Balance                  | Gleichmäßige Verteilung der Klassen (z. B. 50/50) |
| Data Quality                   | Qualität und Verlässlichkeit der Trainingsdaten |
| OpenCV                         | Eine Open-Source-Bibliothek zur Bild- und Videoverarbeitung |

---

## 🎯 Modulüberblick (Bild 1)

- Dieses Python-Programm macht Bilderkennung durch manuelle Annotation möglich.  
- Man könnte sagen: „Wir sortieren Urlaubsfotos für die KI – nur dass die KI später das Sortieren übernimmt.“  
- Ziel: Trainingsdaten für ein **Computer-Vision-Projekt** vorbereiten.  
- Humorvolle Analogie: Die KI ist wie ein Kind – erst muss man ihr zeigen, was ein „Riffbarsch“ ist, bevor sie es allein erkennt.

---

## ⚙️ Workflow im Detail (Bild 2)

- **Schritt 1:** Scannen des Ordners nach Bildern.  
- **Schritt 2:** Laden und Vorverarbeiten der Bilder mit **OpenCV**.  
- **Schritt 3:** Nutzer entscheidet interaktiv → Tastensteuerung.  
- **Schritt 4:** Bilder werden automatisch einsortiert.  
- Erkläransatz: Das ist quasi die „Fließbandarbeit“, bevor die KI ins Spiel kommt.  
- Motivation: Ohne diese Schritte kann kein **überwachtes Lernen (Supervised Learning)** stattfinden.

---

## 🎮 Benutzersteuerung (Bild 3)

- T-Taste = Riffbarsch gefunden 🐟  
- K-Taste = Kein Riffbarsch 🚫🐟  
- Q-Taste = Programm beenden  
- Humorvoller Vergleich: Das Ganze ist wie Tinder für Fische – Swipe Left oder Right, nur eben mit T und K.  
- Wichtig: **Binary Classification** – nur zwei Möglichkeiten.  
- Vorteile:  
  - Bessere **Datenqualität**  
  - Ausgeglichene **Klassenverteilung**  
  - Nutzung von **Domain Expertise** → Mensch hilft der KI.

---

## 📁 Ordnerstruktur (Bild 4)

- Input: `raw/` enthält Originalbilder.  
- Output: `processed/` mit zwei Unterordnern:  
  - `riffbarsch/` → positive Beispiele  
  - `kein_riffbarsch/` → negative Beispiele  
- Das ist die Grundlage für **Ground Truth Generation**.  
- Merksatz: „Ohne gute Daten keine gute KI – Daten sind das Gold des Machine Learnings.“  
- Hinweis: **Qualitätskontrolle ist entscheidend** – schlampiges Sortieren = schlechte KI.

---

# 💬 Abschlussrunde

- Dieses Modul ist der „Vorspeisengang“ für ein neuronales Netz: Wir liefern sortierte, geprüfte und gelabelte Bilder.  
- Daraus entsteht ein **sauberes Trainingsset** → später kann ein **Neuronales Netzwerk** lernen, automatisch zwischen Riffbarsch und Nicht-Riffbarsch zu unterscheiden.  
- Humorvoll: „Die KI ist am Anfang wie ein Praktikant – wenn wir ihm nichts beibringen, macht er nur Unsinn.“  
