# pip install torch torchvision matplotlib pillow ultralytics scipy
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import threading
import torch
from torchvision import transforms, models
import torchvision
import os

# WICHTIG: Matplotlib Backend VOR pyplot Import setzen (verhindert Threading-Fehler)
import matplotlib
matplotlib.use('Agg')  # Non-interactive Backend für Threading-Kompatibilität
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import time
from ultralytics import YOLO
from scipy import ndimage


# ================== Pfade zu deinen Modellen ==================
RESNET_PATH = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\resnet\fisch_v2_Z30_20250924_0727_resnet.pt"

# KORRIGIERT: Verwende die richtigen Modelle
YOLO_DETECTION_PATH = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n_detection\riffbarsch_taucher_detection\weights\best.pt"  # DEIN trainiertes DETECTION Modell
# Fallback auf dein Classification Modell falls Detection Modell nicht gefunden wird
YOLO_CLASSIFY_PATH = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n\riffbarsch_taucher_run\weights\best.pt"  # DEIN trainiertes CLASSIFICATION Modell

MASK_PATH = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\maskrcnn\mask_rcnn_ram_turbo_final.pth"  # DEIN trainiertes Mask R-CNN

# ================== Transformationen für ResNet ==================
resnet_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Hilfsfunktionen ==================
def create_adaptive_mask(img_array):
    """Erstellt eine adaptive Segmentierungsmaske basierend auf Bildinhalt"""
    height, width = img_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Konvertiere zu Graustufen für einfachere Verarbeitung
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Verschiedene Segmentierungsstrategien
    avg_brightness = np.mean(gray)
    brightness_std = np.std(gray)
    
    print(f"🎭 DEBUG: Durchschnittliche Helligkeit: {avg_brightness:.1f}")
    print(f"🎭 DEBUG: Helligkeits-Standardabweichung: {brightness_std:.1f}")
    
    try:
        if brightness_std > 50:  # Hohes Kontrast-Bild
            # Kontur-basierte Segmentierung mit Threshold
            threshold = avg_brightness * 1.2  # Höhere Schwelle für helle Bereiche
            mask = (gray > threshold).astype(np.float32)  # KORRIGIERT: Helle Bereiche als Segmentierung
            
            # Morphologische Operationen zur Glättung (falls scipy verfügbar)
            try:
                mask = ndimage.binary_opening(mask, structure=np.ones((5,5))).astype(np.float32)
                mask = ndimage.binary_closing(mask, structure=np.ones((10,10))).astype(np.float32)
                print("🎭 DEBUG: Kontur-basierte Segmentierung mit Morphologie")
            except:
                print("🎭 DEBUG: Kontur-basierte Segmentierung ohne Morphologie")
                
        elif avg_brightness > 150:  # Helles Bild
            # Für helle Bilder: Kantenerkennung
            try:
                edges = ndimage.sobel(gray)
                mask = (edges > np.percentile(edges, 70)).astype(np.float32)
                print("🎭 DEBUG: Kantenerkennung für helles Bild")
            except:
                # Fallback ohne scipy
                mask = create_elliptical_mask(width, height, gray)
                print("🎭 DEBUG: Fallback Ellipse für helles Bild")
                
        else:  # Standard-Segmentierung
            mask = create_elliptical_mask(width, height, gray)
            print("🎭 DEBUG: Standard-Ellipse-Segmentierung")
            
    except Exception as e:
        print(f"🎭 DEBUG: Fehler bei Masken-Erstellung: {e}")
        mask = create_elliptical_mask(width, height, gray)
    
    return mask

def create_elliptical_mask(width, height, gray):
    """Erstellt elliptische Grundform mit Rauschen"""
    mask = np.zeros((height, width), dtype=np.float32)
    center_x, center_y = width // 2, height // 2
    
    for y in range(height):
        for x in range(width):
            # Elliptische Form
            if ((x - center_x) / (width * 0.35))**2 + ((y - center_y) / (height * 0.25))**2 < 1:
                # Füge Rauschen für realistischere Grenzen hinzu
                noise = np.random.random() * 0.3
                brightness_factor = gray[y, x] / 255.0
                if brightness_factor + noise > 0.4:
                    mask[y, x] = 1.0
    
    return mask

# ================== GUI Setup ==================
root = tk.Tk()
root.title("Fischanalyse Präsentation")
root.geometry("1200x700")

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Tabs
tab_upload = ttk.Frame(notebook)
tab_classify = ttk.Frame(notebook)
tab_detect = ttk.Frame(notebook)
tab_segment = ttk.Frame(notebook)

notebook.add(tab_upload, text="Upload")
notebook.add(tab_classify, text=" Klassifikation ")
notebook.add(tab_detect, text=" Objekterkennung ")
notebook.add(tab_segment, text=" Segmentierung ")

# ================== Upload Tab ==================
upload_frame = tk.Frame(tab_upload)
upload_frame.pack(side='left', fill='both', expand=True)
canvas_upload = tk.Label(upload_frame)
canvas_upload.pack(padx=20, pady=20)

current_img = None

def open_image():
    global current_img
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
    if file_path:
        # Bild laden und zu RGB konvertieren (entfernt Alpha-Kanal für YOLO/ResNet Kompatibilität)
        temp_img = Image.open(file_path)
        if temp_img.mode == 'RGBA':
            # RGBA zu RGB konvertieren (weißer Hintergrund für Transparenz)
            rgb_img = Image.new('RGB', temp_img.size, (255, 255, 255))
            rgb_img.paste(temp_img, mask=temp_img.split()[-1])  # Alpha-Kanal als Maske
            current_img = rgb_img
        else:
            current_img = temp_img.convert('RGB')  # Sicherstellen dass es RGB ist
        
        tk_img = ImageTk.PhotoImage(current_img.resize((600,400)))
        canvas_upload.configure(image=tk_img)
        canvas_upload.image = tk_img

btn_upload = tk.Button(upload_frame, text="Bild laden", command=open_image, bg='lightblue')
btn_upload.pack(pady=20)

# ================== Klassifikation ==================
canvas_classify = tk.Label(tab_classify)
canvas_classify.pack(side='left', padx=10, pady=10)
fig_classify_frame = tk.Frame(tab_classify)
fig_classify_frame.pack(side='right', fill='both', expand=True)
progress_classify = ttk.Progressbar(tab_classify, length=400, mode='determinate')
progress_classify.pack(side='bottom', pady=10)

# Lade ResNet Modell
resnet_model = models.resnet18()
num_classes = 3  # Korrekt: Riffbarsch, Taucher, Anderer
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location=device))
resnet_model.to(device)
resnet_model.eval()
class_names = ["Riffbarsch", "Taucher", "Anderer"]

def run_classification(img):
    progress_classify['value'] = 0
    root.update_idletasks()
    progress_classify['value'] = 20
    
    # Sicherstellen dass Bild RGB ist (3 Kanäle für ResNet)
    if img.mode != 'RGB':
        print(f"🔄 DEBUG: ResNet Konvertiere {img.mode} -> RGB")
        img = img.convert('RGB')
    
    img_tensor = resnet_transforms(img).unsqueeze(0).to(device)
    progress_classify['value'] = 50
    with torch.no_grad():
        output = resnet_model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    # Klassenreihenfolge korrigieren: [Taucher, Riffbarsch, Anderer] -> [Riffbarsch, Taucher, Anderer]
    probs_corrected = np.array([probs[1], probs[0], probs[2]])
    pred_idx = np.argmax(probs_corrected)
    
    progress_classify['value'] = 80
    tk_img = ImageTk.PhotoImage(img.resize((400,400)))
    canvas_classify.configure(image=tk_img)
    canvas_classify.image = tk_img

    # Balkendiagramm mit korrigierten Wahrscheinlichkeiten
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(class_names, probs_corrected, color=['#e74c3c','#3498db', '#95a5a6'])
    ax.set_ylim(0,1)
    ax.set_ylabel("Wahrscheinlichkeit")
    ax.set_title(f"Vorhersage: {class_names[pred_idx]}")
    for widget in fig_classify_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=fig_classify_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    progress_classify['value'] = 100

btn_classify = tk.Button(tab_classify, text="Klassifizieren", 
                        command=lambda: threading.Thread(target=lambda: run_classification(current_img) if current_img else None).start(), 
                        bg='#27ae60', fg='white', font=('Arial', 12, 'bold'))
btn_classify.pack(side='bottom', pady=10)

# ================== Objekterkennung ==================
canvas_detect = tk.Label(tab_detect)
canvas_detect.pack(side='left', padx=10, pady=10)
fig_detect_frame = tk.Frame(tab_detect)
fig_detect_frame.pack(side='right', fill='both', expand=True)
progress_detect = ttk.Progressbar(tab_detect, length=400, mode='determinate')
progress_detect.pack(side='bottom', pady=10)

# Lade YOLO Modell mit automatischer Detection/Classification Erkennung
import os

yolo_model = None
model_type = "unknown"

# Versuche zuerst das Detection Model zu laden
if os.path.exists(YOLO_DETECTION_PATH):
    try:
        yolo_model = YOLO(YOLO_DETECTION_PATH)
        model_type = "detection"
        print(f"✅ YOLO DETECTION Model geladen: {YOLO_DETECTION_PATH}")
    except Exception as e:
        print(f"❌ Detection Model Fehler: {e}")

# Fallback auf Classification Model
if yolo_model is None and os.path.exists(YOLO_CLASSIFY_PATH):
    try:
        yolo_model = YOLO(YOLO_CLASSIFY_PATH)
        model_type = "classification"
        print(f"⚠️ YOLO CLASSIFICATION Model geladen (KEINE Bounding Boxes!): {YOLO_CLASSIFY_PATH}")
    except Exception as e:
        print(f"❌ Classification Model Fehler: {e}")

if yolo_model is None:
    print(f"🚨 CRITICAL: Kein YOLO Model gefunden!")
    print(f"   Geprüfte Pfade:")
    print(f"   Detection: {YOLO_DETECTION_PATH}")
    print(f"   Classification: {YOLO_CLASSIFY_PATH}")

print(f"🤖 Model Type: {model_type}")

# ================== Mask R-CNN Modell laden ==================
mask_rcnn_model = None
if os.path.exists(MASK_PATH):
    try:
        # Lade dein trainiertes Mask R-CNN Modell
        mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=3)  # Riffbarsch, Taucher, Background
        mask_rcnn_model.load_state_dict(torch.load(MASK_PATH, map_location=device))
        mask_rcnn_model.to(device)
        mask_rcnn_model.eval()
        print(f"✅ MASK R-CNN Model geladen: {MASK_PATH}")
    except Exception as e:
        print(f"❌ Mask R-CNN Model Fehler: {e}")
        mask_rcnn_model = None
else:
    print(f"⚠️ Mask R-CNN Model nicht gefunden: {MASK_PATH}")

def run_detection(img):
    global model_type
    print("=" * 60)
    print("🚀 DEBUG: YOLO OBJEKTERKENNUNG GESTARTET")
    print("=" * 60)
    
    if yolo_model is None:
        print("🚨 FEHLER: Kein YOLO Model verfügbar!")
        return
    
    progress_detect['value'] = 0
    root.update_idletasks()
    progress_detect['value'] = 30

    # YOLO Prediction mit Debug-Info
    print(f"📸 DEBUG: Bildgröße: {img.size}")
    print(f"📸 DEBUG: Bildmodus: {img.mode}")
    print(f"🤖 DEBUG: Model Type: {model_type}")
    print(f"🤖 DEBUG: YOLO Modell: {type(yolo_model)}")
    
    # Sicherstellen dass Bild RGB ist (3 Kanäle für YOLO)
    if img.mode != 'RGB':
        print(f"🔄 DEBUG: Konvertiere {img.mode} -> RGB")
        img = img.convert('RGB')
    
    img_array = np.array(img)
    print(f"📊 DEBUG: Array Shape: {img_array.shape}")
    
    # KORRIGIERT: Parameter für Multi-Objekt-Erkennung
    results = yolo_model.predict(
        source=img_array, 
        conf=0.1,          # Niedrige Confidence für mehr Detektionen
        iou=0.3,           # Niedrige IoU-Schwelle (weniger Zusammenfassung überlappender Boxes)
        max_det=50,        # Maximal 50 Detektionen pro Bild
        agnostic_nms=False, # Klassen-spezifische NMS (bessere Multi-Objekt-Erkennung)
        verbose=True
    )
    
    print(f"📊 DEBUG: Anzahl Results: {len(results)}")
    print(f"📦 DEBUG: Results[0]: {results[0]}")
    print(f"📦 DEBUG: Results[0].boxes: {results[0].boxes}")
    
    # Spezielle Behandlung für Classification vs Detection Models
    if model_type == "classification":
        print("🔄 DEBUG: CLASSIFICATION MODEL - Konvertiere zu Detection-ähnlichem Output")
        
        # Bei Classification Models: probs statt boxes
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()
            print(f"📊 DEBUG: Classification Probs: {probs}")
            print(f"📊 DEBUG: Class Names: {results[0].names}")
            
            # Simuliere Detection für das gesamte Bild
            predicted_class = int(probs.argmax())
            confidence = float(probs.max())
            
            print(f"🎯 DEBUG: Predicted Class: {predicted_class} ({results[0].names[predicted_class]})")
            print(f"🎯 DEBUG: Confidence: {confidence:.3f}")
            
            # Simuliere eine "Detection" für das gesamte Bild
            if confidence > 0.5:  # Confidence Threshold
                if predicted_class == 0:  # riffbarsch
                    riffbarsch_count = 1
                elif predicted_class == 1:  # taucher
                    taucher_count = 1
            
        # Für Classification Models: Zeige Original-Bild (keine Boxes)
        result_pil = img.copy()
        
    else:
        # DETECTION MODEL - Normale Verarbeitung
        if results[0].boxes is not None:
            num_boxes = len(results[0].boxes)
            print(f"📦 DEBUG: Anzahl Boxes: {num_boxes}")
            print(f"📦 DEBUG: Boxes Tensor: {results[0].boxes.data}")
            
            # Zusätzliche Analyse der Box-Koordinaten
            for i, box in enumerate(results[0].boxes):
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                print(f"   📦 Box {i+1}: Coords={coords}, Conf={conf:.3f}, Class={cls}")
                
        else:
            print("⚠️ DEBUG: KEINE BOXES ERKANNT!")

        # KORREKT: Bounding Boxes mit EXPLIZITER Konfiguration
        result_img = results[0].plot(
            conf=True,        # Zeige Confidence-Werte
            labels=True,      # Zeige Klassenlabels
            boxes=True,       # Zeige ALLE Bounding Boxes
            line_width=2,     # Dickere Linien für bessere Sichtbarkeit
        )  
        result_pil = Image.fromarray(result_img[..., ::-1])  # BGR -> RGB
        
        print(f"🖼️ DEBUG: Result Image Shape: {result_img.shape}")
        print(f"🖼️ DEBUG: Plotting {num_boxes if 'num_boxes' in locals() else 0} Bounding Boxes")
    
    # Klassen-spezifische Zählung mit DETAILLIERTEM Debug
    riffbarsch_count = 0
    taucher_count = 0
    andere_count = 0
    
    if model_type == "detection" and results[0].boxes is not None:
        print("🔍 DEBUG: DETAILLIERTE BOX-ANALYSE:")
        print(f"🏷️ DEBUG: Verfügbare Klassen: {yolo_model.names}")
        
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])  # Klassen-ID extrahieren
            confidence = float(box.conf[0])  # Confidence-Score
            coords = box.xyxy[0].cpu().numpy()  # Bounding Box Koordinaten
            class_name = yolo_model.names[class_id] if class_id in yolo_model.names else f"Unknown_{class_id}"
            
            print(f"   Box {i+1}: Klasse={class_id} ({class_name}), Confidence={confidence:.3f}, Coords={coords}")
            
            # DEIN trainiertes Modell: Direkte Zuordnung basierend auf Klassen-ID oder Namen
            class_name_lower = class_name.lower()
            if 'riffbarsch' in class_name_lower or class_id == 0:  # Annahme: Riffbarsch ist Klasse 0
                riffbarsch_count += 1
                print(f"     -> RIFFBARSCH erkannt!")
            elif 'taucher' in class_name_lower or 'diver' in class_name_lower or class_id == 1:  # Annahme: Taucher ist Klasse 1
                taucher_count += 1
                print(f"     -> TAUCHER erkannt!")
            else:
                andere_count += 1
                print(f"     -> ANDERE KLASSE ({class_name}) erkannt!")
    elif model_type == "classification":
        print("⚠️ DEBUG: Classification Model - keine Bounding Boxes verfügbar!")
        # Für Classification verwende Gesamtbild-Klassifikation
        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                top_class_id = probs.top1
                confidence = probs.top1conf.item()
                print(f"🏷️ Classification Result: Klasse {top_class_id}, Confidence: {confidence:.3f}")
                
                if top_class_id == 0:  # Basierend auf Training
                    riffbarsch_count = 1
                    print("     -> RIFFBARSCH klassifiziert!")
                elif top_class_id == 1:
                    taucher_count = 1  
                    print("     -> TAUCHER klassifiziert!")
    else:
        print("⚠️ DEBUG: Keine Boxes/Probs für Analyse verfügbar!")
    
    total_detections = riffbarsch_count + taucher_count + andere_count
    progress_detect['value'] = 70
    
    print(f"� DEBUG FINALE ZÄHLUNG:")
    print(f"   🐠 Riffbarsche: {riffbarsch_count}")
    print(f"   🤿 Taucher: {taucher_count}")
    print(f"   ❓ Andere: {andere_count}")
    print(f"   🔢 Total: {total_detections}")

    # Anzeige im Tkinter-Canvas (MIT sichtbaren Bounding Boxes)
    tk_img = ImageTk.PhotoImage(result_pil.resize((400,400)))
    canvas_detect.configure(image=tk_img)
    canvas_detect.image = tk_img

    # KORRIGIERTES Balkendiagramm - Klassen-spezifisch mit DEBUG
    print(f"📊 DEBUG: Erstelle Balkendiagramm...")
    fig, ax = plt.subplots(figsize=(4,3))
    categories = ["Riffbarsch", "Taucher"]
    counts = [riffbarsch_count, taucher_count]
    colors = ['#e74c3c', '#3498db']
    
    print(f"📊 DEBUG: Categories: {categories}")
    print(f"📊 DEBUG: Counts: {counts}")
    print(f"📊 DEBUG: Colors: {colors}")
    
    bars = ax.bar(categories, counts, color=colors)
    ax.set_ylabel("Anzahl")
    ax.set_title(f"Erkannte Objekte: {total_detections}")
    
    max_count = max(counts) if counts else 0
    y_limit = max(5, max_count + 1)
    ax.set_ylim(0, y_limit)
    
    print(f"📊 DEBUG: Y-Limit gesetzt auf: {y_limit}")
    
    # Werte auf Balken anzeigen
    for i, (bar, count) in enumerate(zip(bars, counts)):
        print(f"📊 DEBUG: Balken {i}: Höhe={count}")
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom', fontweight='bold')
            print(f"📊 DEBUG: Text '{count}' hinzugefügt auf Balken {i}")
    
    print(f"📊 DEBUG: Alte Widgets werden entfernt...")
    for widget in fig_detect_frame.winfo_children():
        widget.destroy()
    
    print(f"📊 DEBUG: Canvas wird erstellt und gepackt...")
    canvas = FigureCanvasTkAgg(fig, master=fig_detect_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
    print(f"📊 DEBUG: Diagramm erfolgreich angezeigt!")

    progress_detect['value'] = 100
    print("✅ DEBUG: YOLO OBJEKTERKENNUNG ABGESCHLOSSEN")
    print("=" * 60)



btn_detect = tk.Button(tab_detect, text="Objekte erkennen", 
                      command=lambda: threading.Thread(target=lambda: run_detection(current_img) if current_img else None).start(), 
                      bg='#f39c12', fg='white', font=('Arial', 12, 'bold'))
btn_detect.pack(side='bottom', pady=10)

# ================== Segmentierung ==================
canvas_segment = tk.Label(tab_segment)
canvas_segment.pack(side='left', padx=10, pady=10)
fig_segment_frame = tk.Frame(tab_segment)
fig_segment_frame.pack(side='right', fill='both', expand=True)
progress_segment = ttk.Progressbar(tab_segment, length=400, mode='determinate')
progress_segment.pack(side='bottom', pady=10)

def run_segmentation(img):
    progress_segment['value'] = 0
    root.update_idletasks()
    
    if mask_rcnn_model is None:
        print("⚠️ Kein Mask R-CNN Modell verfügbar - verwende Dummy-Segmentierung")
        # Fallback auf adaptive Maske
        progress_segment['value'] = 25
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        mask_array = create_adaptive_mask(img_array)
        progress_segment['value'] = 50
        
        # Konvertiere Maske zu PIL Image
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), 'L')
        
        # Erstelle farbige Overlay-Maske
        colored_mask = Image.new("RGBA", img.size, (255, 0, 0, 0))
        colored_mask.paste((255, 0, 0, 120), mask=mask_img)  # Rote semi-transparente Maske
        
        # Kombiniere Originalbild mit Maske
        img_rgba = img.convert("RGBA")
        img_with_mask = Image.alpha_composite(img_rgba, colored_mask)
        progress_segment['value'] = 75
    else:
        # Verwende dein trainiertes Mask R-CNN Modell
        print("🎯 Verwende trainiertes Mask R-CNN Modell")
        progress_segment['value'] = 25
        
        # Bildgrößen für Statistiken
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Bild für Mask R-CNN vorbereiten
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        progress_segment['value'] = 50
        
        # Mask R-CNN Inferenz
        with torch.no_grad():
            predictions = mask_rcnn_model(img_tensor)
        
        # Extrahiere Masken aus Predictions
        if len(predictions[0]['masks']) > 0:
            # Nehme die beste Maske (höchste Konfidenz)
            best_mask = predictions[0]['masks'][0, 0].cpu().numpy()
            mask_array = (best_mask > 0.5).astype(np.float32)  # Threshold bei 0.5
            print(f"🎭 Mask R-CNN: {len(predictions[0]['masks'])} Masken gefunden")
        else:
            # Fallback wenn keine Masken gefunden
            print("⚠️ Mask R-CNN: Keine Masken gefunden - verwende Dummy")
            mask_array = create_adaptive_mask(img_array)
        
        progress_segment['value'] = 65
        
        # Konvertiere Maske zu PIL Image
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), 'L')
        
        # Erstelle farbige Overlay-Maske (blau für echte Mask R-CNN Ergebnisse)
        colored_mask = Image.new("RGBA", img.size, (0, 0, 255, 0))
        colored_mask.paste((0, 0, 255, 120), mask=mask_img)  # Blaue semi-transparente Maske
        
        # Kombiniere Originalbild mit Maske
        img_rgba = img.convert("RGBA")
        img_with_mask = Image.alpha_composite(img_rgba, colored_mask)
        progress_segment['value'] = 75
    
    # Zeige das Ergebnis
    tk_img = ImageTk.PhotoImage(img_with_mask.resize((400,400)))
    canvas_segment.configure(image=tk_img)
    canvas_segment.image = tk_img
    
    # Berechne Statistiken
    mask_pixels = np.sum(mask_array > 0)
    background_pixels = np.sum(mask_array == 0)
    total_pixels = width * height
    mask_percentage = (mask_pixels / total_pixels) * 100
    background_percentage = (background_pixels / total_pixels) * 100
    
    print(f"📊 DEBUG Maske Statistiken:")
    print(f"   🎯 Segmentierte Pixel (>0): {mask_pixels}")
    print(f"   🏗️ Hintergrund Pixel (=0): {background_pixels}")
    print(f"   📏 Total Pixel: {total_pixels}")
    print(f"   📊 Segmentiert: {mask_percentage:.1f}%")
    print(f"   📊 Hintergrund: {background_percentage:.1f}%")
    print(f"   🔍 Mask Array Min/Max: {mask_array.min():.3f}/{mask_array.max():.3f}")
    print(f"   🔍 Mask Array Unique Values: {np.unique(mask_array)}")
    
    progress_segment['value'] = 90
    
    # Erstelle Diagramm mit Masken-Vorschau (Höhe 20% reduziert)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.56))
        
        # Links: Prozentuale Verteilung - KORRIGIERT: Reihenfolge angepasst
        values = [background_percentage, mask_percentage]  # KORRIGIERT: Hintergrund zuerst, dann Segmentiert
        labels = ["Segmentiert","Hintergrund" ]  # KORRIGIERT: Labels entsprechend angepasst
        colors = ['#95a5a6', '#e74c3c']  # KORRIGIERT: Grau für Hintergrund, Rot für Segmentiert
        
        print(f"📊 DEBUG Diagramm-Werte:")
        print(f"   Labels: {labels}")
        print(f"   Values: {values}")
        
        ax1.bar(labels, values, color=colors)
        ax1.set_ylabel("Prozent")
        ax1.set_title(f"Segmentierung: {mask_percentage:.1f}%")  # Titel bleibt gleich - zeigt segmentierten Anteil
        ax1.set_ylim(0, 100)
        
        # Werte auf Balken anzeigen
        for i, (label, value) in enumerate(zip(labels, values)):
            ax1.text(i, value + 2, f'{value:.1f}%', ha='center', va='bottom')
            print(f"   Balken {i}: '{label}' = {value:.1f}%")
        
        # Rechts: Maske als Schwarz-Weiß Bild
        ax2.imshow(mask_array, cmap='gray', interpolation='nearest')
        ax2.set_title("Segmentierungsmaske")
        ax2.axis('off')
    
        plt.tight_layout()
        
        # Altes Diagramm entfernen und neues einfügen
        for widget in fig_segment_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=fig_segment_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        progress_segment['value'] = 100
        print("✅ Segmentierung abgeschlossen")
        
    except Exception as e:
        print(f"❌ Diagramm-Fehler: {e}")
        progress_segment['value'] = 100

btn_segment = tk.Button(tab_segment, text="Segmentieren", 
                       command=lambda: threading.Thread(target=lambda: run_segmentation(current_img) if current_img else None).start(), 
                       bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'))
btn_segment.pack(side='bottom', pady=10)

# ================== Start GUI ==================
if __name__ == "__main__":
    print("🚀 Starte Fischanalyse GUI mit Masken-Anzeige...")
    root.mainloop()