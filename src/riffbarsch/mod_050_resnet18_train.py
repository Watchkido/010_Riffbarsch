# pip install torch torchvision albumentations scikit-learn seaborn numpy
import torch, torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import sys
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
import numpy as np

# Transform (nur Standardisierung und Tensor)
train_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Korrekte Pfade für ImageFolder: Die Klassenordner müssen direkt unter train/ val/ test/ liegen
train_dir = r"E:/dev/projekt_python_venv/010_Riffbarsch/datasets/resnet/train"
val_dir   = r"E:/dev/projekt_python_venv/010_Riffbarsch/datasets/resnet/val"
test_dir  = r"E:/dev/projekt_python_venv/010_Riffbarsch/datasets/resnet/test"

train_ds = ImageFolder(train_dir, transform=train_transforms)
val_ds   = ImageFolder(val_dir, transform=val_transforms)
test_ds  = ImageFolder(test_dir, transform=val_transforms)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=14, pin_memory=True)
val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=14, pin_memory=True)
test_dl  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=14, pin_memory=True)

# Modellwahl: resnet18 oder resnet50
MODELL_TYP = "resnet18"  # Alternativ: "resnet50"

if MODELL_TYP == "resnet18":
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
elif MODELL_TYP == "resnet50":
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
else:
    raise ValueError("MODELL_TYP muss 'resnet18' oder 'resnet50' sein!")

# Bestimme die Anzahl der Klassen automatisch aus dem Trainingsdatensatz
num_classes = len(train_ds.classes)
model.fc = nn.Linear(num_ftrs, num_classes)  # num_classes z.B. 3 (riffbarsch, hard_negatives, taucher)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

MAX_TRAINING_MINUTES = 30  # Maximale Trainingszeit in Minuten
start_time = time.time()

MODELLNAME = "fisch"
VERSION = "v2"
LAUFZEIT_MIN = MAX_TRAINING_MINUTES
MODELLORDNER = r"E:/dev/projekt_python_venv/010_Riffbarsch/models/resnet"
os.makedirs(MODELLORDNER, exist_ok=True)

EARLY_STOPPING_PATIENCE = 5  # Anzahl der Epochen ohne Verbesserung bis Abbruch
best_val_loss = float('inf')
epochs_no_improve = 0
all_val_labels = []
all_val_preds = []

if __name__ == "__main__":
    try:
        # Trainingsloop (vereinfacht)
        for epoch in range(1, 31):
            elapsed = (time.time() - start_time) / 60
            print(f"[Fortschritt] Epoche {epoch}/30 | Vergangene Zeit: {elapsed:.1f} min | Zeitlimit: {MAX_TRAINING_MINUTES} min")
            if elapsed > MAX_TRAINING_MINUTES:
                print(f"Training abgebrochen nach {MAX_TRAINING_MINUTES} Minuten.")
                break
            
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for batch_idx, (imgs, labels) in enumerate(train_dl):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                
                # Accuracy berechnen
                _, preds = torch.max(out, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)
                
                if batch_idx % 10 == 0:
                    print(f"  [Train] Batch {batch_idx+1}/{len(train_dl)} | Aktueller Loss: {loss.item():.4f}")
            
            epoch_loss = running_loss / len(train_dl.dataset)
            train_acc = correct_train / total_train
            train_losses.append(epoch_loss)
            train_accuracies.append(train_acc)

            # Validierung
            model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            val_epoch_labels = []
            val_epoch_preds = []
            with torch.no_grad():
                for val_idx, (imgs, labels) in enumerate(val_dl):
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    loss = criterion(out, labels)
                    val_running_loss += loss.item() * imgs.size(0)
                    
                    # Accuracy und Predictions sammeln
                    _, preds = torch.max(out, 1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)
                    val_epoch_labels.extend(labels.cpu().numpy())
                    val_epoch_preds.extend(preds.cpu().numpy())
                    
                    if val_idx % 10 == 0:
                        print(f"  [Val] Batch {val_idx+1}/{len(val_dl)} | Aktueller Loss: {loss.item():.4f}")
            
            val_loss = val_running_loss / len(val_dl.dataset)
            val_acc = correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            all_val_labels = val_epoch_labels
            all_val_preds = val_epoch_preds
            print(f"[Epoche {epoch}] Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            # Early Stopping prüfen
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  [Early Stopping] Keine Verbesserung für {epochs_no_improve} Epoche(n).")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Training gestoppt durch Early Stopping nach {epoch} Epochen (Val Loss hat sich {EARLY_STOPPING_PATIENCE} Epochen nicht verbessert).")
                    break

        # Modell speichern mit Zeitstempel, Laufzeit, Version und Name
        now = datetime.now().strftime("%Y%m%d_%H%M")
        modell_dateiname = f"{MODELLNAME}_{VERSION}_Z{LAUFZEIT_MIN}_{now}_resnet.pt"
        modell_pfad = os.path.join(MODELLORDNER, modell_dateiname)
        torch.save(model.state_dict(), modell_pfad)
        print(f"Modell gespeichert als {modell_pfad}")

        # 1. Loss & Accuracy-Kurven
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.title(f'Loss für: {modell_dateiname}')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoche')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy für: {modell_dateiname}')
        plt.legend()
        plt.grid(True)

        loss_acc_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_loss_acc.png'))
        plt.tight_layout()
        plt.savefig(loss_acc_pfad)
        plt.show()
        print(f"Loss & Accuracy-Kurven gespeichert als {loss_acc_pfad}")

        # 2. Confusion Matrix (Validierung)
        cm = confusion_matrix(all_val_labels, all_val_preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=train_ds.classes, yticklabels=train_ds.classes)
        plt.xlabel('Vorhergesagt')
        plt.ylabel('Tatsächlich')
        plt.title(f'Confusion Matrix für: {modell_dateiname}')
        cm_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_confusion.png'))
        plt.savefig(cm_pfad)
        plt.show()
        print(f"Confusion Matrix gespeichert als {cm_pfad}")

        # 3. ROC-Kurve & AUC (nur für Binärklassifikation)
        if num_classes == 2:
            val_probs = []
            model.eval()
            with torch.no_grad():
                for imgs, _ in val_dl:
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                    val_probs.extend(probs)
            
            fpr, tpr, _ = roc_curve(all_val_labels, val_probs)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            plt.plot([0,1], [0,1], 'k--', label='Zufall')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC-Kurve für: {modell_dateiname}')
            plt.legend()
            plt.grid(True)
            roc_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_roc.png'))
            plt.savefig(roc_pfad)
            plt.show()
            print(f"ROC-Kurve gespeichert als {roc_pfad}")

            # 4. Precision-Recall-Kurve
            precision, recall, _ = precision_recall_curve(all_val_labels, val_probs)
            plt.figure(figsize=(6,5))
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall für: {modell_dateiname}')
            plt.grid(True)
            pr_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_pr.png'))
            plt.savefig(pr_pfad)
            plt.show()
            print(f"Precision-Recall-Kurve gespeichert als {pr_pfad}")

        # 5. Beispielbilder mit Vorhersage
        n_examples = 8
        test_imgs, test_labels = [], []
        for imgs, labels in test_dl:
            test_imgs.extend(imgs)
            test_labels.extend(labels)
            if len(test_imgs) >= n_examples:
                break
        test_imgs = test_imgs[:n_examples]
        test_labels = test_labels[:n_examples]
        
        model.eval()
        fig, axes = plt.subplots(2, 4, figsize=(16,8))
        axes = axes.flatten()
        
        with torch.no_grad():
            for i, ax in enumerate(axes):
                if i < len(test_imgs):
                    img = test_imgs[i]
                    label = test_labels[i].item()
                    img_input = img.unsqueeze(0).to(device)
                    output = model(img_input)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred = np.argmax(probs)
                    
                    # Denormalisierung für Anzeige
                    img_display = img.clone()
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])
                    for t, m, s in zip(img_display, mean, std):
                        t.mul_(s).add_(m)
                    img_display = torch.clamp(img_display, 0, 1)
                    
                    ax.imshow(np.transpose(img_display.numpy(), (1,2,0)))
                    ax.axis('off')
                    color = 'green' if pred == label else 'red'
                    ax.set_title(f"Tatsächlich: {train_ds.classes[label]}\n"
                               f"Vorhersage: {train_ds.classes[pred]}\n"
                               f"Konfidenz: {probs[pred]:.2f}", color=color)
                else:
                    ax.axis('off')
        
        ex_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_examples.png'))
        plt.tight_layout()
        plt.savefig(ex_pfad)
        plt.show()
        print(f"Beispielbilder gespeichert als {ex_pfad}")

        # 5.2. Alle falsch klassifizierten Bilder sammeln und anzeigen
        model.eval()
        falsch_klassifiziert = []
        
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Sammle alle falsch klassifizierten Bilder
                for i in range(len(imgs)):
                    if preds[i] != labels[i]:
                        # Denormalisierung für Anzeige
                        img_display = imgs[i].cpu().clone()
                        mean = torch.tensor([0.485, 0.456, 0.406])
                        std = torch.tensor([0.229, 0.224, 0.225])
                        for t, m, s in zip(img_display, mean, std):
                            t.mul_(s).add_(m)
                        img_display = torch.clamp(img_display, 0, 1)
                        
                        falsch_klassifiziert.append({
                            'img': img_display,
                            'true_label': labels[i].cpu().item(),
                            'pred_label': preds[i].cpu().item(),
                            'confidence': probs[i][preds[i]].cpu().item(),
                            'true_class': train_ds.classes[labels[i].cpu().item()],
                            'pred_class': train_ds.classes[preds[i].cpu().item()]
                        })
        
        # Zeige falsch klassifizierte Bilder an (maximal 20)
        if falsch_klassifiziert:
            n_falsch = min(len(falsch_klassifiziert), 20)
            cols = 5
            rows = (n_falsch + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for i in range(len(axes)):
                if i < n_falsch:
                    item = falsch_klassifiziert[i]
                    img_display = np.transpose(item['img'].numpy(), (1,2,0))
                    
                    axes[i].imshow(img_display)
                    axes[i].axis('off')
                    
                    # Titel mit Farbe je nach Fehlertyp
                    title = f"Tatsächlich: {item['true_class']}\nVorhergesagt: {item['pred_class']}\nKonfidenz: {item['confidence']:.2f}"
                    axes[i].set_title(title, color='red', fontsize=10)
                    
                    # Rahmen um das Bild
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
                else:
                    axes[i].axis('off')
            
            plt.suptitle(f'Falsch klassifizierte Bilder ({len(falsch_klassifiziert)} gefunden, {n_falsch} angezeigt)', 
                        fontsize=16, color='red')
            falsch_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_falsch_klassifiziert.png'))
            plt.tight_layout()
            plt.savefig(falsch_pfad, bbox_inches='tight')
            plt.show()
            print(f"Falsch klassifizierte Bilder gespeichert als {falsch_pfad}")
            print(f"Insgesamt {len(falsch_klassifiziert)} falsch klassifizierte Bilder gefunden.")
        else:
            print("Keine falsch klassifizierten Bilder gefunden - perfekte Klassifikation!")

        # 5.3. Fehlerstatistik nach Klassen
        if falsch_klassifiziert:
            print("\n=== Fehlerstatistik ===")
            for true_class_idx, true_class_name in enumerate(train_ds.classes):
                for pred_class_idx, pred_class_name in enumerate(train_ds.classes):
                    if true_class_idx != pred_class_idx:
                        count = sum(1 for item in falsch_klassifiziert 
                                  if item['true_label'] == true_class_idx and item['pred_label'] == pred_class_idx)
                        if count > 0:
                            print(f"{true_class_name} fälschlicherweise als {pred_class_name} klassifiziert: {count} Bilder")

        # 6. Grad-CAM (Bonus, nur für ResNet18/50)
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            
            target_layer = model.layer4[-1]
            cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
            
            fig, axes = plt.subplots(2, 4, figsize=(16,8))
            axes = axes.flatten()
            
            for i, ax in enumerate(axes):
                if i < len(test_imgs):
                    img = test_imgs[i]
                    label = test_labels[i].item()
                    input_tensor = img.unsqueeze(0).to(device)
                    
                    # Denormalisierung für Grad-CAM
                    img_rgb = img.clone()
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])
                    for t, m, s in zip(img_rgb, mean, std):
                        t.mul_(s).add_(m)
                    img_rgb = torch.clamp(img_rgb, 0, 1)
                    rgb_img = np.transpose(img_rgb.numpy(), (1,2,0))
                    
                    grayscale_cam = cam(input_tensor=input_tensor)[0]
                    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    
                    ax.imshow(cam_img)
                    ax.axis('off')
                    ax.set_title(f"Grad-CAM: {train_ds.classes[label]}")
                else:
                    ax.axis('off')
            
            cam_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_gradcam.png'))
            plt.tight_layout()
            plt.savefig(cam_pfad)
            plt.show()
            print(f"Grad-CAM gespeichert als {cam_pfad}")
        except ImportError:
            print("Grad-CAM konnte nicht erzeugt werden (pytorch-grad-cam nicht installiert).")
            print("Installiere mit: pip install grad-cam")
            
        # 5.4. Bilder: Als 'riffbarsch' vorhergesagt, aber tatsächlich 'hard_negativ'
        falsch_riffbarsch = [item for item in falsch_klassifiziert 
                            if item['pred_class'] == 'riffbarsch' and item['true_class'] == 'hard_negatives']
        
        if falsch_riffbarsch:
            n_falsch_riffbarsch = min(len(falsch_riffbarsch), 25)
            cols = 5
            rows = (n_falsch_riffbarsch + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for i in range(len(axes)):
                if i < n_falsch_riffbarsch:
                    item = falsch_riffbarsch[i]
                    img_display = np.transpose(item['img'].numpy(), (1,2,0))
                    
                    axes[i].imshow(img_display)
                    axes[i].axis('off')
                    
                    title = f"Tatsächlich: {item['true_class']}\nVorhergesagt: {item['pred_class']}\nKonfidenz: {item['confidence']:.2f}"
                    axes[i].set_title(title, color='orange', fontsize=10)
                    
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor('orange')
                        spine.set_linewidth(2)
                else:
                    axes[i].axis('off')
            
            plt.suptitle(f'Falsch als "riffbarsch" klassifiziert ({len(falsch_riffbarsch)} gefunden, {n_falsch_riffbarsch} angezeigt)', 
                        fontsize=16, color='orange')
            falsch_riffbarsch_pfad = os.path.join(MODELLORDNER, modell_dateiname.replace('.pt', '_falsch_riffbarsch.png'))
            plt.tight_layout()
            plt.savefig(falsch_riffbarsch_pfad, bbox_inches='tight')
            plt.show()
            print(f"Falsch als 'riffbarsch' klassifizierte Bilder gespeichert als {falsch_riffbarsch_pfad}")
        else:
            print("Keine Bilder gefunden, die fälschlicherweise als 'riffbarsch' klassifiziert wurden.")
            
    except KeyboardInterrupt:
        print("Manueller Abbruch durch Strg+C. Das Programm wird beendet.")
    finally:
        sys.exit(0)