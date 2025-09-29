"""
projekt_analyse.py
Dieses Modul bietet Funktionen zur Analyse von Python-Projekten hinsichtlich ihrer Datei- und Importstruktur sowie zur Durchführung einer Flake8-Codeprüfung.

Changelog:
- 2025-01-24: Bereinigte Version - Entfernte 3 doppelte/ungenutzte schreibe_baum Funktionen
  * schreibe_python_dateien_baum_alle() - war ungenutzt
  * schreibe_python_dateien_baum() - war ungenutzt  
  * schreibe_kompletten_verzeichnisbaum() - war ungenutzt
- TODO: Code weiter optimieren und redundante Funktionen vermeiden

Funktionen:
-----------
- finde_python_dateien(root):
    Durchsucht rekursiv ein Verzeichnis nach allen Python-Dateien (.py) und gibt deren Pfade zurück.
- print_verwendete_module(verwendet_von):
    Gibt eine strukturierte Übersicht darüber aus, welche Module von welchen Dateien im Projekt verwendet werden.
- extrahiere_imports(dateiPfad):
    Extrahiert alle importierten Module aus einer gegebenen Python-Datei und gibt diese als Set zurück.
- analysiere_imports(py_dateien):
    Analysiert die Importbeziehungen zwischen den gefundenen Python-Dateien, ermittelt verwendete und nicht verwendete Module und gibt entsprechende Zuordnungen zurück.
- flake8_pruefen(dateien):
    Führt eine Flake8-Codeanalyse für eine Liste von Python-Dateien durch und gibt die Ergebnisse aus.
- hauptfunktion(startverzeichnis):
    Hauptfunktion zur Durchführung der Analyse: Findet alle Python-Dateien, analysiert die Importe, listet nicht verwendete Dateien auf, speichert die Ergebnisse in einer Datei und führt eine Flake8-Prüfung durch.

Verwendung:
-----------
Das Skript kann direkt ausgeführt werden. Es verwendet einen Basis-Pfad aus einer Konfigurationsdatei (CONFIG.BASIS_Pfad) als Startpunkt für die Analyse.

Abhängigkeiten:
---------------
- os
- re
- subprocess
- collections.defaultdict
- config.CONFIG (externe Konfigurationsdatei)
- flake8 (muss installiert sein)

Ergebnis:
---------
Die Analyseergebnisse werden sowohl auf der Konsole ausgegeben als auch in der Datei 'import_analyse_ergebnis.txt' gespeichert.
"""
import os
import re
import subprocess
from collections import defaultdict

import ast
from config import CONFIG

# -------------------------------------------
# 🔎 Datei- & Import-Analyse (bereinigt)
# -------------------------------------------

def finde_python_dateien(root: str) -> dict:
    """
    Durchsucht rekursiv den angegebenen Projektordner und alle Unterordner (z. B. data, notebooks, prompts)
    nach Python-Dateien, ignoriert aber das .venv-Verzeichnis.

    :param root: Startverzeichnis (Projektordner)
    :type root: str
    :return: Baumstruktur aller gefundenen .py-Dateien (als dict)
    :rtype: dict
    :example:
        >>> finde_python_dateien("meinprojekt")
        {'main.py': None, 'data': {'dataset.py': None}, 'notebooks': {'auswertung.py': None}}
    """
    def baum(Pfad):
        """
        Gibt alle Python-Dateien als verschachtelte Baumstruktur (geschachtelte Dicts) zurück.
        
        Beispiel:
        {
            'ordner1': {
                'datei1.py': None,
                'unterordner': {
                    'datei2.py': None
                }
            },
            'datei3.py': None
        }
        
        :param Pfad: Startverzeichnis
        :type Pfad: str
        :return: Baumstruktur als dict
        :rtype: dict
        """
        baum_dict = {}
        try:
            eintraege = sorted(os.listdir(Pfad))
        except PermissionError:
            return baum_dict
        for eintrag in eintraege:
            vollPfad = os.path.join(Pfad, eintrag)
            if eintrag in (".venv", ".git", "__pycache__"):
                continue
            if os.path.isdir(vollPfad):
                unterbaum = baum(vollPfad)
                if unterbaum:
                    baum_dict[eintrag] = unterbaum
            elif eintrag.endswith(".py"):
                baum_dict[eintrag] = None
        return baum_dict
    return baum(root)


def print_verwendete_module(verwendet_von):
    """
    Gibt eine strukturierte Übersicht darüber aus, welche Module von welchen Dateien verwendet werden.
    Baut einen Baum zur besseren Visualisierung der Modulabhängigkeiten.
    
    :param verwendet_von: Dictionary mit Modul -> Liste der verwendenden Dateien
    :type verwendet_von: dict
    """
    print("\n🔗 \033[1mVerwendete Module:\033[0m")
    print("".ljust(60, "─"))
    for modul, verwendet_durch in verwendet_von.items():
        print(f"📦 \033[94m{modul}\033[0m")
        # Gruppiere nach Wurzelverzeichnis
        baum = {}
        for Pfad in verwendet_durch:
            teile = Pfad.split(os.sep)
            d = baum
            for teil in teile:
                d = d.setdefault(teil, {})
        
        def print_baum(d, prefix="  "):
            """
            Druckt Baumstruktur auf der Konsole aus
            
            :param d: Dictionary mit Baumstruktur
            :type d: dict
            :param prefix: Einrückung für die Darstellung  
            :type prefix: str
            """
            for i, (name, sub) in enumerate(d.items()):
                connector = "└── " if i == len(d)-1 else "├── "
                print(prefix + connector + name)
                if sub:
                    print_baum(sub, prefix + ("    " if i == len(d)-1 else "│   "))
        
        print_baum(baum)
        print("".ljust(60, "─"))


# ENTFERNT: 3 doppelte/ungenutzte schreibe_baum Funktionen:
# - schreibe_python_dateien_baum_alle() (Zeile 118-148)
# - schreibe_python_dateien_baum() (Zeile 153-191) 
# - schreibe_kompletten_verzeichnisbaum() (Zeile 212-248)


def extrahiere_imports(dateiPfad):
    """
    Extrahiert alle importierten Module aus einer Python-Datei.
    
    :param dateiPfad: Pfad zur Python-Datei
    :type dateiPfad: str
    :return: Set der importierten Modulnamen
    :rtype: set
    :raises FileNotFoundError: Wenn Datei nicht gefunden wird
    :raises SyntaxError: Wenn Python-Syntax fehlerhaft ist
    """
    imports = set()
    try:
        with open(dateiPfad, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=dateiPfad)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except (FileNotFoundError, UnicodeDecodeError, SyntaxError):
        pass
    return imports


def analysiere_imports(py_dateien):
    """
    Analysiert die Importbeziehungen zwischen den gefundenen Python-Dateien.
    Gibt eine Zuordnung von Modulnamen zu Dateien, eine Übersicht der verwendeten Module
    und eine Liste nicht verwendeter Dateien zurück.
    
    :param py_dateien: Liste aller Python-Dateien
    :type py_dateien: list
    :return: Tuple aus (modulnamen_to_dateien, verwendet_von, nicht_verwendet)
    :rtype: tuple
    """
    modulnamen_to_dateien = {os.path.splitext(os.path.basename(f))[0]: f for f in py_dateien}
    verwendet_von = defaultdict(list)
    
    for datei in py_dateien:
        imports = extrahiere_imports(datei)
        dateiname = os.path.splitext(os.path.basename(datei))[0]
        
        for imp in imports:
            if imp in modulnamen_to_dateien and imp != dateiname:
                verwendet_von[imp].append(dateiname)
    
    # Nicht verwendete Dateien finden
    verwendete_module = set(verwendet_von.keys())
    alle_module = set(modulnamen_to_dateien.keys())
    nicht_verwendet = [modulnamen_to_dateien[m] for m in alle_module - verwendete_module 
                      if not m.startswith('__') and not m.startswith('test_')]
    
    return modulnamen_to_dateien, dict(verwendet_von), nicht_verwendet


def finde_und_liste_alle_funktionen():
    """
    Findet alle Funktionen in allen Python-Dateien des Projekts
    
    :returns: Dictionary mit Dateipfad -> Liste der Funktionen
    :rtype: dict
    """
    funktionen_dict = {}
    
    def baum_zu_liste(baum, basis):
        """
        Wandelt Baumstruktur in flache Liste von Dateipfaden um
        
        :param baum: Baumstruktur als Dictionary
        :type baum: dict
        :param basis: Basispfad
        :type basis: str
        :returns: Liste der Dateipfade
        :rtype: list
        """
        Pfade = []
        for name, sub in baum.items():
            Pfad = os.path.join(basis, name)
            if sub is None:
                Pfade.append(Pfad)
            else:
                Pfade.extend(baum_zu_liste(sub, Pfad))
        return Pfade
    
    # Alle Python-Dateien finden
    py_baum = finde_python_dateien(CONFIG["PROJEKT_Pfad"])
    py_dateien = baum_zu_liste(py_baum, CONFIG["PROJEKT_Pfad"])
    
    for datei_pfad in py_dateien:
        funktionen_dict[datei_pfad] = []
        try:
            with open(datei_pfad, 'r', encoding='utf-8') as f:
                baum = ast.parse(f.read())
                
            for knoten in ast.walk(baum):
                if isinstance(knoten, ast.FunctionDef):
                    funktionen_dict[datei_pfad].append(knoten.name)
                    
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            funktionen_dict[datei_pfad] = ["Fehler beim Parsen der Datei"]
    
    return funktionen_dict


def flake8_pruefen(dateien):
    """
    Führt eine Flake8-Prüfung für die angegebenen Dateien durch.
    
    :param dateien: Liste von Dateipfaden
    :type dateien: list
    """
    print("\n🧹 \033[1mFlake8-Prüfung:\033[0m")
    print("".ljust(60, "─"))
    
    if not dateien:
        print("❌ Keine Dateien zum Prüfen gefunden.")
        return
    
    try:
        # Flake8 mit erweiterten Regeln ausführen
        result = subprocess.run(['flake8'] + dateien, 
                              capture_output=True, text=True, 
                              cwd=os.path.dirname(dateien[0]))
        
        if result.returncode == 0:
            print("✅ Alle Dateien entsprechen den Flake8-Standards!")
        else:
            print("❌ Flake8 hat Probleme gefunden:")
            print(result.stdout)
            if result.stderr:
                print("Fehlerausgabe:", result.stderr)
                
    except FileNotFoundError:
        print("❌ Flake8 ist nicht installiert. Installiere es mit: pip install flake8")
    except Exception as e:
        print(f"❌ Fehler bei der Flake8-Prüfung: {e}")



def erstelle_ascii_baum(baum, prefix="", ist_letzter=True):
    """
    Erstellt eine ASCII-Baumdarstellung der Projektstruktur
    
    :param baum: Baumstruktur als Dictionary
    :type baum: dict
    :param prefix: Prefix für die aktuelle Einrückung
    :type prefix: str
    :param ist_letzter: Ob dies der letzte Eintrag auf dieser Ebene ist
    :type ist_letzter: bool
    :return: ASCII-Baum als String
    :rtype: str
    """
    result = ""
    items = list(baum.items())
    
    for i, (name, sub) in enumerate(items):
        ist_letzter_item = (i == len(items) - 1)
        
        # Symbol für Datei/Ordner bestimmen
        if sub is None:  # Datei
            if name.endswith('.py'):
                symbol = "📄"
            elif name.endswith('.md'):
                symbol = "📄"
            elif name.endswith('.txt'):
                symbol = "📄"
            else:
                symbol = "📄"
        else:  # Ordner
            symbol = "📁"
        
        # Verbinder bestimmen
        connector = "└── " if ist_letzter_item else "├── "
        
        # Zeile hinzufügen
        result += f"{prefix}{connector}{symbol} {name}\n"
        
        # Wenn es ein Ordner ist, rekursiv weitermachen
        if sub is not None:
            new_prefix = prefix + ("    " if ist_letzter_item else "│   ")
            result += erstelle_ascii_baum(sub, new_prefix, ist_letzter_item)
    
    return result


def erstelle_vollstaendigen_projektbaum(startverzeichnis):
    """
    Erstellt einen vollständigen Projektbaum (alle Dateien, nicht nur Python)
    
    :param startverzeichnis: Startverzeichnis
    :type startverzeichnis: str
    :return: Vollständiger Projektbaum
    :rtype: dict
    """
    def baum_vollstaendig(pfad):
        baum_dict = {}
        try:
            eintraege = sorted(os.listdir(pfad))
        except PermissionError:
            return baum_dict
        
        for eintrag in eintraege:
            vollpfad = os.path.join(pfad, eintrag)
            
            # Bestimmte Ordner/Dateien ignorieren
            if eintrag in (".venv", ".git", "__pycache__", ".pytest_cache", "node_modules"):
                continue
            if eintrag.startswith('.') and eintrag not in ('.gitignore', '.env'):
                continue
                
            if os.path.isdir(vollpfad):
                unterbaum = baum_vollstaendig(vollpfad)
                # Nur hinzufügen wenn Unterbaum nicht leer oder wichtiger Ordner
                if unterbaum or eintrag in ('src', 'data', 'models', 'docs', 'tests', 'scripts', 'notebooks'):
                    baum_dict[eintrag] = unterbaum if unterbaum else {}
            else:
                # Wichtige Dateien immer hinzufügen
                if (eintrag.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json', '.cfg', '.ini')) or 
                    eintrag in ('requirements.txt', 'LICENSE', 'setup.py', 'Dockerfile')):
                    baum_dict[eintrag] = None
        
        return baum_dict
    
    return baum_vollstaendig(startverzeichnis)

# 🧰 Hauptfunktion
# -------------------------------------------

def hauptfunktion(startverzeichnis: str) -> None:
    """
    Führt die komplette Analyse durch:
    - Erstellt ASCII-Baumdarstellung der Projektstruktur
    - Findet alle Python-Dateien
    - Analysiert die Importe
    - Gibt verwendete und nicht verwendete Module aus
    - Speichert die Ergebnisse in einer Datei (mit Baumstruktur)
    - Führt eine Flake8-Prüfung durch
    - Listet alle Funktionen pro Datei auf

    :param startverzeichnis: Startverzeichnis für die Analyse
    :type startverzeichnis: str
    """
    print(f"🔍 Analyse im Projektordner: {startverzeichnis}\n")

    # Vollständige Projektstruktur erstellen
    projekt_baum = erstelle_vollstaendigen_projektbaum(startverzeichnis)
    projekt_name = os.path.basename(startverzeichnis)
    
    # ASCII-Baum erstellen
    ascii_baum = f"{projekt_name}/\n"
    ascii_baum += erstelle_ascii_baum(projekt_baum)

    # Alle Python-Dateien im Projekt finden (als Baumstruktur)
    py_baum = finde_python_dateien(startverzeichnis)

    # Hilfsfunktion: Baumstruktur in flache Liste von DateiPfaden umwandeln
    def baum_zu_liste(baum, basis):
        """
        Wandelt Baumstruktur in Liste von Dateipfaden um
        
        :param baum: Baumstruktur als Dictionary
        :type baum: dict  
        :param basis: Basispfad
        :type basis: str
        :returns: Liste der Dateipfade
        :rtype: list
        """
        Pfade = []
        for name, sub in baum.items():
            Pfad = os.path.join(basis, name)
            if sub is None:
                Pfade.append(Pfad)
            else:
                Pfade.extend(baum_zu_liste(sub, Pfad))
        return Pfade

    py_dateien = baum_zu_liste(py_baum, startverzeichnis)

    # Importbeziehungen analysieren
    alle_module, verwendet_von, nicht_genutzt = analysiere_imports(py_dateien)

    # Ausgabe der Projektstruktur (Konsole)
    print("🏗️ Projektstruktur:")
    print("".ljust(80, "─"))
    print(ascii_baum)

    # Ausgabe der gefundenen Dateien (Konsole)
    print("📄 Gefundene Python-Dateien:")
    for modul, Pfad in alle_module.items():
        print(f"  {modul:<20} → {Pfad}")

    # Ausgabe der verwendeten Module (Konsole)
    print("\n🔗 Importierte Module (aus Projekt):")
    for modul, verwendet_durch in verwendet_von.items():
        print(f"  {modul:<20} verwendet in:")
        for nutzer in verwendet_durch:
            print(f"     └── {nutzer}")

    # Ausgabe nicht verwendeter Dateien (Konsole)
    print(f"\n🧹 Nicht verwendete Dateien ({len(nicht_genutzt)}):")
    for Pfad in nicht_genutzt:
        print(f"  ❌ {os.path.basename(Pfad)} → {Pfad}")

    # Funktionen pro Datei finden
    funktionen_dict = finde_und_liste_alle_funktionen()

    # Ergebnisse in Datei speichern (MIT ASCII-BAUM AM ANFANG!)
    with open("import_analyse_ergebnis.txt", "w", encoding="utf-8") as f:
        # ASCII-Projektstruktur am Anfang
        f.write("🏗️ PROJEKTSTRUKTUR\n")
        f.write("=" * 80 + "\n")
        f.write(ascii_baum)
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Rest der Analyse
        f.write("📄 Python-Dateien:\n")
        f.write("-" * 40 + "\n")
        for modul, Pfad in alle_module.items():
            f.write(f"{modul} → {Pfad}\n")
        
        f.write("\n🔗 Verwendete Module:\n")
        f.write("-" * 40 + "\n")
        for modul, verwendet_durch in verwendet_von.items():
            f.write(f"{modul} verwendet in:\n")
            for nutzer in verwendet_durch:
                f.write(f"  └── {nutzer}\n")
        
        f.write("\n🧹 Nicht verwendete Dateien:\n")
        f.write("-" * 40 + "\n")
        for Pfad in nicht_genutzt:
            f.write(f"❌ {Pfad}\n")
        
        f.write("\n📝 Funktionen pro Datei:\n")
        f.write("-" * 40 + "\n")
        for Pfad, funktionen in funktionen_dict.items():
            f.write(f"{Pfad}:\n")
            for name in funktionen:
                f.write(f"  - {name}()\n")

    flake8_pruefen(py_dateien)
    print("\n✅ Analyse abgeschlossen. Ergebnisse gespeichert in 'import_analyse_ergebnis.txt'")
    print(f"📁 Vollständige Projektstruktur mit {len(projekt_baum)} Hauptelementen erstellt.")
# 🏁 Ausführung
# -------------------------------------------

if __name__ == "__main__":
    # Startet die Analyse mit dem in der Konfiguration hinterlegten Basis-Pfad
    hauptfunktion(CONFIG["PROJEKT_Pfad"])
    #print_verwendete_module(verwendet_von)