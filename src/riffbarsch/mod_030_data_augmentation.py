import os
import cv2
import albumentations as A
from datetime import datetime
# Basisordner immer abändern zu den quell und zielordnern!

input_dir = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\downloads\reef_fish_striped_school_underwater"
base_output = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\processed\i_net_data_augmentation"

# original: 
# 010_Riffbarsch\datasets\downloads\Abudefduf_fish_group_reef


# Transformationen systematisch definieren
transforms = {
    "gespiegelt": A.Compose([A.HorizontalFlip(p=1.0)]),
    "gespiegelt_vertikal": A.Compose([A.VerticalFlip(p=1.0)]),
    "gelöchert": A.Compose([
        A.CoarseDropout(max_holes=2, p=1.0)
    ]),
    "blur": A.Compose([A.GaussianBlur(blur_limit=(5, 7), p=1.0)]),
    "hell_dunkel": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)]),
    "farbe": A.Compose([A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=1.0)]),
    "crop": A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.75, 1.33), p=1.0)
    ]),
    "shift_scale": A.Compose([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=1.0)]),
    "perspektive": A.Compose([A.Perspective(scale=(0.05,0.1), p=1.0)]),
    "distortion": A.Compose([A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=1.0)]),
    "rauschen": A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)]),
    "clahe": A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=1.0)]),
    "grid_distortion": A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)]),
    "elastic": A.Compose([A.ElasticTransform(alpha=50, sigma=5, alpha_affine=20, p=1.0)]),
    "posterize": A.Compose([A.Posterize(num_bits=3, p=1.0)]),
    "superpixel": A.Compose([A.Superpixels(p_replace=0.1, n_segments=200, p=1.0)])
}


# Zusätzliche Rotationen in festen Winkeln
rotation_angles = [10, 20, 30,40,50, 60,70,80]
for angle in rotation_angles:
    transforms[f"rotiert_{angle}grad"] = A.Compose([A.Rotate(limit=(angle, angle), p=1.0)])

# Zielordner anlegen
for name in transforms.keys():
    os.makedirs(os.path.join(base_output, name), exist_ok=True)

# Bilder durchgehen und alle Varianten erzeugen
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filepath = os.path.join(input_dir, filename)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for name, transform in transforms.items():
            augmented = transform(image=image)["image"]

            out_dir = os.path.join(base_output, name)
            # Dateiname mit Zeitstempel
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_{now}{ext}"
            out_path = os.path.join(out_dir, new_filename)

            cv2.imwrite(out_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

print("✅ Alle Transformationen fertig! Sie liegen in Unterordnern von:")
print(base_output)
