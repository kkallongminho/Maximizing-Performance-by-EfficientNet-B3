#베이스라인
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os

# ✅ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 경로
train_dir = "/kaggle/input/deepfake-and-real-images/Dataset/Train"
val_dir = "/kaggle/input/deepfake-and-real-images/Dataset/Validation"

# ✅ 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ✅ EfficientNet-B3 불러오기
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# ✅ 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ✅ 모델 저장 경로
save_path = "/kaggle/working/efficientnet_b3_best_full.pth"
os.makedirs("/kaggle/working", exist_ok=True)

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

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # ✅ 검증 단계
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

        # ✅ F1-score 기준 베스트 모델 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model, save_path)
            print(f"🔥 Best model saved at {save_path} (F1: {best_f1:.4f})")

    print(f"\n🎯 Training complete. Best Val F1: {best_f1:.4f}")

# ✅ 학습 실행
train_model(model, train_loader, val_loader, epochs=10)
