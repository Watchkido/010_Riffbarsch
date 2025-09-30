Präsentation: Datenaufteilung für Neuronale Netze - Die drei spezialisierten Dataset-Partitionierungs-Module
🎯 Fach- und Fremdwörter
Fremdwort	Deutsche Erklärung
Dataset-Partitionierungspipelines	Automatisierte Prozesse zur Aufteilung von Datensätzen in Trainings-, Validierungs- und Testdaten.
Neuronale Netze	KI-Modelle, die inspiriert sind vom menschlichen Gehirn und Muster in Daten erkennen lernen.
ResNet	Ein tiefes neuronales Netzwerk, das speziell für Bildklassifikation entwickelt wurde.
YOLOv8	Ein ultraschnelles Objekterkennungsmodell für Echtzeitanwendungen.
Mask R-CNN	Ein Modell für Instanzsegmentierung, das Objekte pixelgenau umrandet.
Stratified Sampling	Eine Aufteilungsmethode, die die Klassenverteilung in allen Datensätzen proportional gleich hält.
Multi-Source Collection	Das Sammeln von Daten aus mehreren Quellordnern für einen diverseren Datensatz.
Instance Segmentation	Pixelgenaue Erkennung und Abgrenzung einzelner Objekte in einem Bild.
Binary Classification	Die Unterscheidung zwischen genau zwei Klassen (z.B. "Ja/Nein" oder "Riffbarsch/Taucher").
Random Shuffling	Das zufällige Mischen von Daten, um Verzerrungen durch die Reihenfolge zu vermeiden.
🎨 Präsentations-Stichpunkte
Slide 1: Übersicht der drei Module (Bild 040_1.png)
🎯 DIE DREI DATEN-SORTIERER

Drei spezialisierte Python-Module für verschiedene KI-Modelle

Jedes Modul = ein spezialisierter Sortierroboter

ResNet: Sortiert in 3 Kategorien wie ein gut organisierter Büroangestellter

YOLOv8: Blitzschnelle 2-Klassen-Erkennung wie ein Formel-1-Pilot

Mask R-CNN: Pixelgenaue Umrandung wie ein hochpräziser Chirurg

💬 Sprechtext:
"Stellt euch vor, ihr habt tausende Fotos von Unterwasseraufnahmen und wollt daraus KI-Modelle trainieren. Aber halt! Bevor die KI lernen kann, müssen wir die Daten richtig sortieren. Und genau dafür haben wir unsere drei spezialisierten Daten-Sortierer entwickelt - jeder ein Meister seines Fachs!"

Slide 2: Gemeinsamkeiten & Unterschiede (Bild 040_2.png)
📈 WAS VERBINDET UNSERE MODULE?

🎯 Gleiche Aufteilung: 70% Training, 15% Validation, 15% Test

🔄 Stratified Sampling: Jede Klasse wird fair verteilt

📁 Multi-Source: Bilder aus mehreren Quellordnern

🤖 Vollautomatisch: Ein Klick, alles erledigt

🎲 Random Shuffling: Keine langweiligen Muster

💬 Sprechtext:
"Was haben unsere drei Module gemeinsam? Sie sind wie eine gut organisierte Familie - sie teilen fair auf, mischen gründlich durch und arbeiten vollautomatisch. Aber wie in jeder Familie hat jedes Mitglied seine eigenen Spezialitäten. Wer von euch hat schonmal erlebt, dass falsch aufgeteilte Daten zu schlechten Modellen geführt haben?"

Slide 3: Drei-Modul-Vergleich (Bild 040_3.png)
📊 UNSER DREIERGESPANN IM DETAIL

🐠 ResNet (Modul 040) - Der Präzisionsarbeiter

3 Klassen: Riffbarsch, Taucher, Hard Negatives

Für hohe Genauigkeit bei komplexen Klassifikationen

Wie ein Wissenschaftler: gründlich, genau, zuverlässig

⚡ YOLOv8 (Modul 060) - Der Geschwindigkeitsdämon

2 Klassen: Riffbarsch vs. Taucher

Blitzschnell für Echtzeitanwendungen

Wie ein Sportler: schnell, effizient, immer bereit

🔍 Mask R-CNN (Modul 080) - Der Präzisionskünstler

2 Klassen mit pixelgenauer Umrandung

Zeigt nicht nur WAS, sondern auch WO im Bild

Wie ein Künstler: detailverliebt, präzise, kreativ

💬 Sprechtext:
"Hier seht ihr unser Dream-Team! ResNet ist unser gründlicher Wissenschaftler, YOLOv8 der blitzschnelle Sportler und Mask R-CNN der detailverliebte Künstler. Jeder hat seine Stärken - die Frage ist nur: Welcher passt zu eurem Projekt?"

Slide 4: Ordnerstruktur-Vergleich (Bild 040_4.png)
📁 WIE SIE ALLES ORDNEN

ResNet-Struktur:

text
train/val/test/
├── riffbarsch/
├── hard_negatives/  
└── taucher/
YOLOv8-Struktur:

text
yolo_classification/
├── train/val/test/
│   ├── riffbarsch/
│   └── taucher/
Mask R-CNN-Struktur:

text
maskrcnn_data/
├── train/val/test/
│   ├── riffbarsch/timestamp_images.jpg
│   └── taucher/timestamp_images.jpg
💬 Sprechtext:
"Und das Beste: Jedes Modul erstellt die perfekte Ordnerstruktur für das jeweilige Framework. Kein manuelles Sortieren, kein Herumprobieren - einfach laufen lassen und die KI ist glücklich! Das ist wie wenn jemand euren Kleiderschrank perfekt organisiert, nur für KI-Modelle!"

Slide 5: Anwendungsempfehlungen (Bild 040_5.png)
🎯 WELCHES MODUL FÜR WELCHE AUFGABE?

🐠 ResNet wählen wenn:

Ihr mehrere Klassen unterscheiden müsst

Genauigkeit wichtiger ist als Geschwindigkeit

Ihr wissenschaftliche Forschung betreibt

⚡ YOLOv8 wählen wenn:

Geschwindigkeit das A und O ist

Das Modell auf mobilen Geräten laufen soll

Ein einfaches Ja/Nein-Problem gelöst werden muss

🔍 Mask R-CNN wählen wenn:

Ihr pixelgenaue Umrandungen braucht

Nicht nur WAS, sondern auch WO wichtig ist

Höchste Präzision gefragt ist

💬 Sprechtext:
"Und jetzt die große Frage: Welches Modul solltet ihr wählen? Ganz einfach: Für die meisten Anwendungen ist YOLOv8 der Star - schnell, effizient und zuverlässig. Für komplexe Klassifikationen vertraut auf ResNet. Und wenn ihr pixelgenaue Analysen braucht, ist Mask R-CNN euer bester Freund. Denkt dran: Das richtige Werkzeug für die richtige Aufgabe - das ist der Schlüssel zum Erfolg!"

🎤 Präsentations-Tipps
Einstieg:
"Hallo zusammen! Heute zeige ich euch, wie wir aus einem Chaos von Bildern perfekt organisierte Trainingsdaten machen - mit unseren drei spezialisierten Daten-Sortierern!"

Zwischendurch:
"Das ist wie wenn ihr drei verschiedene Köche für verschiedene Gerichte habt - jeder ist ein Meister in seiner Küche!"

Abschluss:
"Mit diesen drei Modulen habt ihr für jede KI-Aufgabe die perfekte Datenaufteilung parat. Egal ob schnell, präzise oder pixelgenau - wir haben die Lösung!"

Interaktion:
"Ich bin neugierig - welches der drei Modelle würdet ihr für eure nächste KI-Anwendung wählen? Schnell wie YOLO, präzise wie ResNet oder detailverliebt wie Mask R-CNN?"

Motivation:
"Vergesst manuelles Sortieren - mit unseren automatisierten Pipelines könnt ihr euch auf das Wesentliche konzentrieren: Tolle KI-Modelle entwickeln!"

