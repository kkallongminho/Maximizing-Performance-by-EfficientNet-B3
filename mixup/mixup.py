#mixup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os
from timm import create_model
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "/kaggle/input/deepfake-and-real-images/Dataset/Train"
val_dir = "/kaggle/input/deepfake-and-real-images/Dataset/Validation"

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ✅ EfficientNet-B3 (DropPath 제거)
model = create_model('tf_efficientnet_b3', pretrained=True, num_classes=2, drop_path_rate=0.0)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

save_path = "/kaggle/working/efficientnet_b3_mixup_best.pth"
os.makedirs("/kaggle/working", exist_ok=True)

# ✅ MixUp 함수
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ✅ 학습 함수
def train_model(model, train_loader, val_loader, epochs=10):
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        print(f"\n[Epoch {epoch+1}/{epochs}] Training...")
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            mixed_images, y_a, y_b, lam = mixup_data(images, labels)
            outputs = model(mixed_images)
            loss = mixup_criterion(outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_bar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * images.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f"Epoch {epoch+1}/{epochs} ✅")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model, save_path)
            print(f"🔥 Best model saved at {save_path} (F1: {best_f1:.4f})")

    print(f"\n🎯 Training complete. Best Val F1: {best_f1:.4f}")

train_model(model, train_loader, val_loader, epochs=10)

