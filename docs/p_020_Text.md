# 🎤 Präsentations-Skript: Hard Negative Mining

Hey Team, lasst uns über ein kleines Abenteuer in der Welt der neuronalen Netze reden!  
Stellt euch vor, ihr bringt jemandem bei, echte Designer-Handtaschen zu erkennen – würdet ihr nur die echten zeigen? Neeeein! Ihr würdet auch die besten Fälschungen zeigen. Genau das macht Hard Negative Mining mit Bildern für unser KI-Modell.

Und jetzt gehen wir gemeinsam die Abschnitte durch:


## 📚 Fach- und Fremdwörter

| Fremdwort | Deutsche Erklärung |
|-----------|--------------------|
| Hard Negative Mining | Gezieltes Sammeln von Bildern, die ähnlich aussehen wie Zielobjekte, aber keine sind. |
| Adversarial Data Collection Pipeline | Automatischer Prozess, um schwierige Beispiele zu finden. |
| Adversarial Training | Trainingsmethode, bei der man absichtlich schwierige Fälle zeigt. |
| Decision Boundary Refinement | Das Modell lernt genauere Grenzen zwischen Klassen zu ziehen. |
| False Positive Reduction | Weniger Fehlalarme bei der Klassifikation. |
| Model Generalization | Das Modell funktioniert besser auf neuen, unbekannten Daten. |
| Robustness Training | Das Modell wird widerstandsfähiger gegen Verwechslungen. |
| Domain Adaptation | Das Modell kann auch in neuen Umgebungen eingesetzt werden. |
| icrawler | Python-Bibliothek für Web-Bildersuche. |
| requests | Bibliothek für HTTP-Anfragen in Python. |
| urllib3 | Bibliothek für sicheres Laden von URLs. |
| random/time | Python-Module zur Simulation menschlichen Verhaltens. |
| BingImageCrawler | Crawler aus icrawler, um Bilder über Bing zu sammeln. |

### 🎯 Modulüberblick
- Hard Negative Mining = schwierige Negativbeispiele sammeln.
- Wie bei gefälschten Designer-Handtaschen: man muss auch die guten Fakes zeigen.
- Das Skript: automatisches Sammeln ähnlicher, aber falscher Bilder.

### 🧠 Strategische Kategorien
- Vier Hauptbereiche: Unterwasser, Landmuster, künstliche Objekte, Sci-Fi.
- Beispiele: Zebras, Wespen, Fantasy-Monster.
- Ziel: Modell verwirren – damit es später weniger verwirrt ist 😉.

### ⚙️ Technischer Workflow
- 1. Suchbegriffe generieren (40+).
- 2. Web Scraping mit BingImageCrawler.
- 3. Batch-Download (50 Bilder/Begriff).
- 4. Pausen wie ein Mensch einbauen (2–5 Sek).
- Sinn: klare Entscheidungslinien für das Modell.

### 📊 Technische Spezifikationen
- 40+ Suchbegriffe, 2000+ Bilder, 7 Kategorien.
- Libraries: icrawler, requests, urllib3, random/time.
- Python-Code zeigt, wie Bilder gesammelt und gespeichert werden.

### 💡 Machine Learning Vorteile
- Weniger False Positives, bessere Generalisierung.
- Modell versteht: was ist wirklich kein Riffbarsch.
- Qualitätskontrolle bleibt wichtig: Müllbilder rauswerfen.

---

## 🚀 Schlusswort

Das war unsere Reise durch das Hard Negative Mining Modul.  
Am Ende bedeutet das: unser Modell lernt nicht nur das Offensichtliche, sondern auch die kniffligen Details.  
Oder kurz gesagt: Wir machen unsere KI **street-smart** – nicht nur **book-smart** 😎.

