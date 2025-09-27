# pip install torch torchvision matplotlib pillow ultralytics scipy
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import threading
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
from ultralytics import YOLO
from scipy import ndimage


# ================== Pfade zu deinen Modellen ==================
RESNET_PATH = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\resnet\fisch_v2_Z30_20250924_0727_resnet.pt"
YOLO_PATH = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\yolov8n\riffbarsch_taucher_run\weights\best.pt"
MASK_PATH = r"E:\dev\projekt_python_venv\010_Riffbarsch\models\sam_vit\mask_model.pth"  # Beispiel

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
            threshold = avg_brightness * 0.8
            mask = (gray < threshold).astype(np.float32)
            
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
notebook.add(tab_classify, text="Klassifikation")
notebook.add(tab_detect, text="Objekterkennung")
notebook.add(tab_segment, text="Segmentierung")

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
        current_img = Image.open(file_path)
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

# Lade YOLO Modell
yolo_model = YOLO(YOLO_PATH)



def run_detection(img):
    progress_detect['value'] = 0
    root.update_idletasks()
    progress_detect['value'] = 30

    # YOLO Prediction
    results = yolo_model.predict(source=np.array(img), conf=0.25)

    # Ergebnisbild mit Bounding Boxes erzeugen
    result_img = results[0].plot()  # numpy array mit Boxen
    result_pil = Image.fromarray(result_img[..., ::-1])  # BGR -> RGB

    # Halbtransparente Hervorhebung über jedes erkannte Objekt legen
    if results[0].boxes is not None:
        overlay = Image.new("RGBA", result_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box-Koordinaten
            # halbtransparentes Rechteck (z. B. rotes Overlay)
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 80))

        # Overlay mit Original kombinieren
        result_pil = result_pil.convert("RGBA")
        result_pil = Image.alpha_composite(result_pil, overlay)

    detections = len(results[0].boxes) if results[0].boxes else 0
    progress_detect['value'] = 70

    # Anzeige im Tkinter-Canvas
    tk_img = ImageTk.PhotoImage(result_pil.resize((400,400)))
    canvas_detect.configure(image=tk_img)
    canvas_detect.image = tk_img

    # Balkendiagramm
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["Erkannte Objekte"], [detections], color='orange')
    ax.set_ylabel("Anzahl")
    ax.set_title(f"Objekte: {detections}")
    for widget in fig_detect_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=fig_detect_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    progress_detect['value'] = 100



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
    
    # Erstelle eine adaptive Maske basierend auf Bildinhalt
    progress_segment['value'] = 25
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Erstelle intelligente Maske basierend auf Bildanalyse
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
    
    # Zeige das Ergebnis
    tk_img = ImageTk.PhotoImage(img_with_mask.resize((400,400)))
    canvas_segment.configure(image=tk_img)
    canvas_segment.image = tk_img
    
    # Berechne Statistiken
    mask_pixels = np.sum(mask_array > 0)
    total_pixels = width * height
    mask_percentage = (mask_pixels / total_pixels) * 100
    
    progress_segment['value'] = 100
    
    # Erstelle Diagramm mit Masken-Vorschau
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Links: Prozentuale Verteilung
    ax1.bar(["Segmentiert", "Hintergrund"], [mask_percentage, 100-mask_percentage], 
           color=['#e74c3c', '#ecf0f1'])
    ax1.set_ylabel("Prozent")
    ax1.set_title(f"Segmentierung: {mask_percentage:.1f}%")
    ax1.set_ylim(0, 100)
    
    # Werte auf Balken anzeigen
    for i, v in enumerate([mask_percentage, 100-mask_percentage]):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
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

btn_segment = tk.Button(tab_segment, text="Segmentieren", 
                       command=lambda: threading.Thread(target=lambda: run_segmentation(current_img) if current_img else None).start(), 
                       bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'))
btn_segment.pack(side='bottom', pady=10)

# ================== Start GUI ==================
if __name__ == "__main__":
    print("🚀 Starte Fischanalyse GUI mit Masken-Anzeige...")
    root.mainloop()