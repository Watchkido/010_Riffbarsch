🐠 Von Riff-Fotos zur KI-Anwendung: Eine Reise in die Data Science
Eine Präsentation für angehende Data Analysts
1. Die Urlaubs-App, die nicht existierte
Hallo zusammen! Stellt euch vor: Ihr seid im Urlaub, habt hunderte Fotos von bunten Fischen und Tauchern im Riff gemacht. Ihr wollt die schönsten Momente finden, aber das Durchklicken ist mühsam. „Es wäre doch cool, wenn eine App alle Fische und Taucher automatisch erkennt“, dachte ich mir. Aber so eine App gab es nicht. Und genau hier beginnt die Reise für uns als Data Analysts.

Die Herausforderung war klar: Ich wollte ein Machine-Learning-Programm bauen, das genau das kann. Es sollte nicht nur erkennen, was im Bild ist, sondern auch, wo es ist. Klingt nach Zauberei? Lasst uns das Geheimnis lüften.

2. Der 3-Schritte-Workflow: Vom Foto zur Vorhersage
Wie trainiert man eine Maschine, Fische zu sehen? Der Prozess ist in drei einfache Schritte unterteilt:

Schritt 1: Die Daten-Pipeline – Unsere Trainingsgrundlage

Schritt 2: Das Modell-Training – Unsere Gehirn-OP

Schritt 3: Die Inferenz – Unsere Magie in Aktion

Dieser Workflow ist das Herzstück fast jedes Machine-Learning-Projekts.

3. Die Daten-Pipeline: Der Weg ist das Ziel
Bevor die KI etwas lernen kann, braucht sie einen Lehrer: Daten. Meine 500 Urlaubsfotos waren der Anfang. Doch das reicht nicht.

Annotation: Ich musste dem Programm sagen, was ein Fisch ist und was ein Taucher. Dazu zeichnet man von Hand sogenannte Bounding Boxes (umschließende Rechtecke) um jedes Objekt. Das Ergebnis ist ein riesiger Ordner mit Bildern und zugehörigen Koordinaten. Das sind unsere Trainingsdaten.

Feature Engineering: Das ist die Kunst, aus Rohdaten verwertbare Informationen zu gewinnen. In meinem Fall war das die Datenaugmentation. Ich habe jedes Bild kopiert, gedreht, gespiegelt, aufgehellt oder die Farben verändert. So wurden aus 500 Bildern über 5.000 Trainingsdaten! Mein Modell lernt so, Fische aus jedem Winkel und bei jedem Licht zu erkennen.

Analogie: Das ist wie im E-Sport. Ein Team im EVE Online lernt nicht nur, feindliche Raumschiffe in der Standard-Konfiguration zu identifizieren, sondern auch bei schlechter Sicht, in Bewegung oder mit veränderten Skins. Je diverser die Übung, desto besser die Performance.

4. Das Modell-Training: Das Gehirn unserer App
Nach der Vorbereitung kommt das Training. Ich habe ein neuronales Netzwerk namens YOLOv8 (You Only Look Once, Version 8) verwendet.  Dieses Modell ist besonders schnell, weil es in nur einem Schritt die Objekte erkennt.

Der Trainingsprozess ist eine Iteration:

Das Modell bekommt ein Bild und die dazugehörigen Trainingsdaten gezeigt.

Es macht eine Vorhersage.

Die Verlustfunktion (Loss Function) misst, wie groß der Fehler zwischen Vorhersage und Realität ist. Ein hoher Wert bedeutet: "Du liegst falsch!"

Der Optimierer (Optimizer) passt die internen Gewichte des Modells an, um den Fehler zu minimieren.

Metapher: Stell dir das vor wie ein Kind, das lernt, ein Bild von einem Hund zu zeichnen. Zuerst sieht die Zeichnung wie ein Kritzeltier aus (hoher Loss). Mit jedem Korrekturhinweis (der Optimizer) verbessert es die Linien und Formen, bis der Hund erkennbar ist (niedriger Loss).

Nach 100 Epochen (Durchläufen durch den gesamten Datensatz) war mein Modell bereit.

5. Anwendung in der Praxis: Über Fische hinaus
Jetzt, wo wir die Technik verstehen, schauen wir auf die Relevanz für euch als angehende Data Analysts.

Frauenrechte & Chancengleichheit: KI-Modelle werden in der Kreditvergabe zur Risikobewertung eingesetzt. Analysieren wir Daten nach features wie Einkommen, Zahlungshistorie, etc., können wir faire, unvoreingenommene Entscheidungen treffen und sicherstellen, dass Frauen, die früher benachteiligt wurden, die gleichen Chancen auf einen Kredit haben.

Flucht & Migration: Hilfsorganisationen nutzen Datenanalyse, um die Verteilung von Nahrungsressourcen in Krisengebieten zu optimieren. Satellitenbilder können mit Objekterkennung analysiert werden, um Flüchtlingscamps, landwirtschaftliche Flächen oder sogar Wasservorkommen zu identifizieren und die Hilfe effizienter zu gestalten.

Datenanalyse ist kein Selbstzweck, sondern ein Werkzeug, um reale, komplexe Probleme zu lösen und die Welt ein Stück besser zu machen.

6. Fazit & Dein Call-to-Action
Machine Learning ist keine Magie, sondern ein strukturierter Prozess: Daten sammeln, aufbereiten, ein Modell trainieren und dann die Ergebnisse bewerten.

Du hast gelernt, dass Trainingsdaten das Fundament sind.

Du hast verstanden, wie ein Modell durch eine Verlustfunktion und einen Optimierer lernt.

Und du hast gesehen, dass diese Fähigkeiten weit über Urlaubsbilder hinaus gehen – sie können von der humanitären Hilfe bis zur fairen Kreditvergabe überall eingesetzt werden.

Meine App gibt es jetzt! Und ich habe sie mit meinen Urlaubsfotos gebaut.  Jetzt seid ihr dran.

Dein Weg zum Data Analyst beginnt nicht mit komplizierten Formeln, sondern mit einer einfachen Frage: Welches Problem willst du lösen?

Probier es selbst! Such dir ein kleines Problem, sammle Daten und fang an zu experimentieren. Es muss nicht perfekt sein, es muss dich nur weiterbringen. Und wenn du Hilfe brauchst – die Community ist groß.





Erklärung für die Gruppe
Titel: "Die Schatzsuche des KI-Modells: Eine Reise durch die Verlustlandschaft"

1. Das Ziel der Schatzsuche (Die Mission des Trainings)

Einfach erklärt: Stell dir vor, unser Mask R-CNN-Modell ist ein Roboter, der lernen soll, in Bildern Gegenstände zu finden und genau zu ummalen. Unser Ziel ist es, den besten, klügsten Roboter zu bauen. Der "Schatz" ist die beste Version unseres Roboters, die die wenigsten Fehler macht.

Fachbegriff: Training. Das ist der Prozess, bei dem wir dem Modell zeigen, was richtig und was falsch ist, damit es sich verbessert.

2. Der "Fehler-Zähler" oder die Landkarte (Die Verlustfunktion)

Einfach erklärt: Wie wissen wir, ob unser Roboter gut ist? Wir brauchen einen Bewertungsmaßstab! Das ist die Verlustfunktion (oder Loss Function). Stell sie dir wie eine Punkteliste vor. Jedes Mal, wenn der Roboter einen Fehler macht (z.B. einen Apfel als Birne erkennt oder die Umrandung unsauber zieht), gibt es Minuspunkte. Unser Ziel ist es, so wenig Minuspunkte wie möglich zu sammeln. Je niedriger die Punktzahl, desto besser!

3. Die 3D-Landschaft (Die Loss Landscape)

Einfach erklärt: Jetzt kommt die Grafik! Das ist keine normale Landkarte, sondern eine "Fehler-Landschaft".

Jeder Punkt auf dieser Karte stellt eine andere "Einstellung" oder eine andere Version unseres Roboters dar.

Die Höhe (die z-Achse, die nach oben geht) zeigt, wie viele Fehler diese Version macht. Ein hoher Berg bedeutet viele Fehler. Ein tiefes Tal bedeutet wenige Fehler.

Unser Schatz liegt also irgendwo im tiefsten Tal dieser Landschaft!

4. Die Reise durch die Landschaft (Der Trainingsverlauf über 10 Epochen)

Einfach erklärt: Das Training ist wie eine Wanderung durch diese bergige Landschaft. Unser Roboter startet an einem zufälligen Ort – wahrscheinlich auf einem hohen Berg, wo er noch viele Fehler macht (das ist der hohe, rote/gelbe Bereich zu Beginn).

Fachbegriff: Epoche. Eine Epoche ist wie ein "Lerntag". An einem Lerntag sieht sich der Robotor alle Trainingsbilder einmal an und lernt daraus. Deine Grafik zeigt eine Reise von 10 Lerntagen.

Was passiert bei der Wanderung? Der Roboter geht Schritt für Schritt bergab. Er sucht sich immer die Route, die am steilsten nach unten führt. Nach jedem Schritt (bzw. nach jedem Lerntag/Epoche) gucken wir auf der Karte nach, wo er steht und wie tief er schon gekommen ist.

5. Was uns die Grafik jetzt sagt (Interpretation)

Der Abwärtstrend: Die wichtigste Beobachtung ist, dass die Linie/Landschaft über die Epochen hinweg abwärts geht. Das ist ein sehr gutes Zeichen! Es bedeutet, dass unser Roboter tatsächlich lernt. Er macht von Epoche zu Epoche weniger Fehler. Die Trainingsschatzsuche funktioniert.

Das Ziel ist fast erreicht: Du siehst wahrscheinlich, dass die Kurve nach 10 Epochen flacher wird. Das bedeutet, unser Roboter ist vielleicht schon in der Nähe eines Tals angekommen. Es geht nicht mehr so steil bergab, weil er schon sehr gut geworden ist und nur noch kleine Verbesserungen möglich sind.

Fachbegriff: Konvergenz. Wenn die Kurve ganz flach wird und sich kaum noch verbessert, sagt man, das Modell "konvergiert". Es hat ein sehr gutes Tal gefunden.

Zusammenfassung für die Data Analysts in Ausbildung:
"Die 3D-Verlustlandschaft visualisiert die Performance unseres Mask R-CNN-Modells im Parameterraum über 10 Trainingsepochen. Die z-Achse repräsentiert den Loss-Wert. Der deutliche Abfall der Landschaft zeigt eine erfolgreiche Minimierung der Verlustfunktion an, was auf ein effektives Training hindeutet. Die abflachende Tendenz gegen Ende der Epochen legt nahe, dass das Modell in Richtung Konvergenz geht. Nächste Schritte wären zu prüfen, ob eine längere Trainingsdauer (mehr Epochen) den Loss weiter signifikant senkt oder ob wir Anzeichen von Overfitting beobachten."

Frage an deine Gruppe: "Wenn ihr jetzt die Grafik seht, könnt ihr den Weg des Roboters von einem hohen, fehlerhaften Berg hinab in ein tiefes, fehlerarmes Tal nachvollziehen?"

Diese Erklärung verbindet die intuitive Vorstellung einer Reise mit den präzisen technischen Konzepten, die hinter dem Training eines neuronalen Netzes stehen.

