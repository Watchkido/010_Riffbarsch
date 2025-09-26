import os
import cv2
import shutil

# Ordnerpfade
source_folder = "data/raw"          # hier liegen deine Originalbilder
riffbarsch_folder = "data/processed/riffbarsch"
kein_riffbarsch_folder = "data/processed/kein_riffbarsch"

# Zielordner anlegen, falls nicht vorhanden
os.makedirs(riffbarsch_folder, exist_ok=True)
os.makedirs(kein_riffbarsch_folder, exist_ok=True)

# Alle Fotos laden
extensions = (".jpg", ".jpeg", ".png", ".dng", ".JPG", ".DNG")
images = [f for f in os.listdir(source_folder) if f.endswith(extensions)]

print(f"Gefundene Bilder: {len(images)}")

for img_name in images:
    img_path = os.path.join(source_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Konnte {img_name} nicht laden.")
        continue

    # Bild kleiner anzeigen (sonst riesig)
    img_resized = cv2.resize(img, (800, 600))
    cv2.imshow("Bild - (t = Taucher, k = Kein taucher, q = Beenden)", img_resized)

    key = cv2.waitKey(0)

    if key == ord("t"):  # riffbarsch/taucher
        shutil.move(img_path, os.path.join(riffbarsch_folder, img_name))
    elif key == ord("k"):  # kein riffbarsch
        shutil.move(img_path, os.path.join(kein_riffbarsch_folder, img_name))
    elif key == ord("q"):  # quit
        print("Beendet.")
        break

cv2.destroyAllWindows()
