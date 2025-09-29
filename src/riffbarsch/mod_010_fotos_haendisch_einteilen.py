import os
import cv2
import shutil

# Ordnerpfade - korrigiert um in JPG Unterordner zu schauen
source_folder = "datasets/raw/JPG"    # hier liegen deine JPG Bilder
riffbarsch_folder = "datasets/processed/riffbarsch"
kein_riffbarsch_folder = "datasets/processed/kein_riffbarsch"

# Prüfe ob Quellordner existiert
if not os.path.exists(source_folder):
    print(f"❌ Quellordner '{source_folder}' nicht gefunden!")
    print("Verfügbare Ordner:")
    if os.path.exists("datasets/raw"):
        print(f"  - {os.listdir('datasets/raw')}")
    exit(1)

# Zielordner anlegen, falls nicht vorhanden
os.makedirs(riffbarsch_folder, exist_ok=True)
os.makedirs(kein_riffbarsch_folder, exist_ok=True)

# Alle Fotos laden
extensions = (".jpg", ".jpeg", ".png", ".dng", ".JPG", ".DNG")
images = [f for f in os.listdir(source_folder) if f.endswith(extensions)]

print(f"Gefundene Bilder: {len(images)}")

if len(images) == 0:
    print("❌ Keine Bilder gefunden!")
    print(f"Prüfe Ordner: {os.path.abspath(source_folder)}")
    exit(1)

# OpenCV GUI verfügbarkeit prüfen
try:
    cv2.namedWindow("test")
    cv2.destroyWindow("test")
    gui_available = True
except cv2.error:
    gui_available = False

if not gui_available:
    print("❌ OpenCV GUI nicht verfügbar!")
    print("Mögliche Lösungen:")
    print("1. Installiere opencv-python mit GUI-Support:")
    print("   pip uninstall opencv-python")
    print("   pip install opencv-python-headless==False opencv-python")
    print("2. Oder verwende alternative Bibliothek (PIL/Pillow mit tkinter)")
    exit(1)

print(f"✅ Starte Klassifikation von {len(images)} Bildern...")
print("Tastenbefehle:")
print("  T = Riffbarsch/Taucher (→ riffbarsch/)")  
print("  K = Kein Riffbarsch (→ kein_riffbarsch/)")
print("  Q = Beenden")

for i, img_name in enumerate(images):
    img_path = os.path.join(source_folder, img_name)
    
    # Fortschritt anzeigen
    print(f"\n[{i+1}/{len(images)}] Aktuelles Bild: {img_name}")
    
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Konnte {img_name} nicht laden - überspringe.")
        continue

    # Bild kleiner anzeigen (sonst riesig)
    height, width = img.shape[:2]
    if width > 800 or height > 600:
        scale = min(800/width, 600/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = cv2.resize(img, (new_width, new_height))
    else:
        img_resized = img.copy()
    
    cv2.imshow("Bild klassifizieren - (T=Riffbarsch, K=Kein Riffbarsch, Q=Beenden)", img_resized)

    while True:
        key = cv2.waitKey(0) & 0xFF  # Mask für bessere Kompatibilität

        if key == ord("t") or key == ord("T"):  # riffbarsch/taucher
            dest_path = os.path.join(riffbarsch_folder, img_name)
            shutil.move(img_path, dest_path)
            print(f"✅ → {riffbarsch_folder}/{img_name}")
            break
        elif key == ord("k") or key == ord("K"):  # kein riffbarsch
            dest_path = os.path.join(kein_riffbarsch_folder, img_name)
            shutil.move(img_path, dest_path)
            print(f"✅ → {kein_riffbarsch_folder}/{img_name}")
            break
        elif key == ord("q") or key == ord("Q"):  # quit
            print("🛑 Beendet auf Benutzerwunsch.")
            cv2.destroyAllWindows()
            exit(0)
        else:
            print(f"⚠️ Ungültige Taste '{chr(key) if 32 <= key <= 126 else key}'. Verwende T, K oder Q.")

print(f"✅ Klassifikation abgeschlossen! {len(images)} Bilder verarbeitet.")
cv2.destroyAllWindows()
