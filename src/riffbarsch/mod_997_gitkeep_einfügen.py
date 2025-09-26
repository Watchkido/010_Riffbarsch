# damit dei ordnerstruktur auf git übertragen wird aber nicht die bilder



import os

def add_gitkeep(root_dirs):
    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            gitkeep_path = os.path.join(dirpath, ".gitkeep")
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, "w") as f:
                    pass  # Leere Datei erstellen

# Ordnerpfade anpassen, falls nötig
ordner = [
    r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets"
]

add_gitkeep(ordner)
print("Alle .gitkeep-Dateien wurden hinzugefügt.")