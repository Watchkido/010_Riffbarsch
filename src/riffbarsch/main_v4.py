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

# ================== GUI Setup ==================
root = tk.Tk()
root.title("Fischanalyse Präsentation")
root.geometry("1400x800")

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

# ================== Klassifikation (ResNet) ==================
canvas_classify = tk.Label(tab_classify)
canvas_classify.pack(side='left', padx=10, pady=10)
fig_classify_frame = tk.Frame(tab_classify)
fig_classify_frame.pack(side='right', fill='both', expand=True)
progress_classify = ttk.Progressbar(tab_classify, length=400, mode='determinate')
progress_classify.pack(side='bottom', pady=10)

resnet_model = models.resnet18()
num_classes = 3  # Riffbarsch, Taucher, Anderer
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

    # Reihenfolge korrigieren
    probs_corrected = np.array([probs[1], probs[0], probs[2]])
    pred_idx = np.argmax(probs_corrected)

    progress_classify['value'] = 80
    tk_img = ImageTk.PhotoImage(img.resize((400,400)))
    canvas_classify.configure(image=tk_img)
    canvas_classify.image = tk_img

    # Balkendiagramm mit Wahrscheinlichkeiten
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

# ================== Objekterkennung (YOLO) ==================
canvas_detect = tk.Label(tab_detect)
canvas_detect.pack(side='left', padx=10, pady=10)
fig_detect_frame = tk.Frame(tab_detect)
fig_detect_frame.pack(side='right', fill='both', expand=True)
progress_detect = ttk.Progressbar(tab_detect, length=400, mode='determinate')
progress_detect.pack(side='bottom', pady=10)

yolo_model = YOLO(YOLO_PATH)

def run_detection(img):
    progress_detect['value'] = 0
    root.update_idletasks()
    progress_detect['value'] = 30

    results = yolo_model.predict(source=np.array(img), conf=0.25)

    # Ergebnis mit Boxen
    result_img = results[0].plot()
    result_pil = Image.fromarray(result_img[..., ::-1])

    # Overlay für Hervorhebung
    if results[0].boxes is not None:
        overlay = Image.new("RGBA", result_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        confidences = []
        labels = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            confidences.append(conf)
            labels.append(label)
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 80))
        result_pil = result_pil.convert("RGBA")
        result_pil = Image.alpha_composite(result_pil, overlay)
    else:
        confidences = []
        labels = []

    detections = len(results[0].boxes) if results[0].boxes else 0
    progress_detect['value'] = 70

    # Anzeige im Tkinter
    tk_img = ImageTk.PhotoImage(result_pil.resize((400,400)))
    canvas_detect.configure(image=tk_img)
    canvas_detect.image = tk_img

    # Mehrere Diagramme
    fig, axs = plt.subplots(1, 3, figsize=(12,3))

    # Anzahl erkannter Objekte
    axs[0].bar(["Objekte"], [detections], color='orange')
    axs[0].set_title("Erkannte Objekte")

    # Confidence Scores
    if confidences:
        axs[1].hist(confidences, bins=5, color='green', alpha=0.7)
        axs[1].set_title("Confidence Scores")
        axs[1].set_xlabel("Confidence")
        axs[1].set_xlim(0,1)

    # Klassenverteilung
    if labels:
        unique, counts = np.unique(labels, return_counts=True)
        axs[2].bar(unique, counts, color='blue')
        axs[2].set_title("Klassenhäufigkeit")

    plt.tight_layout()

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

# ================== Segmentierung (Platzhalter) ==================
canvas_segment = tk.Label(tab_segment)
canvas_segment.pack(side='left', padx=10, pady=10)

# (Segmentierungslogik bleibt gleich)

# ================== Start GUI ==================
if __name__ == "__main__":
    print("🚀 Starte Fischanalyse GUI mit Bounding Box + Overlay + Diagrammen...")
    root.mainloop()
