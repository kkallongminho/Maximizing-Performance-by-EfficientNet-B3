#mcdropout
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

# ✅ EfficientNet-B3 불러오기 + Dropout 항상 활성화
class MCDropoutEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)

    # ✅ Dropout 항상 활성화
    def train(self, mode=True):
        super().train(mode)
        # Dropout 강제로 항상 켜기
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(True)

model = MCDropoutEfficientNet().to(device)

# ✅ 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ✅ 모델 저장 경로
save_path = "/kaggle/working/efficientnet_b3_mcdo_best.pth"
os.makedirs("/kaggle/working", exist_ok=True)

# ✅ MCDropout 기반 Validation 함수
def mc_dropout_predict(model, images, mc_samples=5):
    model.train()  # Dropout 유지
    probs = []
    with torch.no_grad():
        for _ in range(mc_samples):
            outputs = model(images)
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    return np.mean(probs, axis=0)  # ✅ 평균 확률 반환

# ✅ 학습 함수
def train_model(model, train_loader, val_loader, epochs=10, mc_samples=5):
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

        # ✅ MCDropout 기반 검증 단계
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_bar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                mc_probs = mc_dropout_predict(model, images, mc_samples=mc_samples)
                preds = np.argmax(mc_probs, axis=1)

                # 손실 계산 (평균 확률 기반)
                outputs_tensor = torch.tensor(mc_probs).to(device)
                loss = criterion(outputs_tensor, labels)

                val_loss += loss.item() * images.size(0)
                all_preds.extend(preds)
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
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Best model saved at {save_path} (F1: {best_f1:.4f})")

    print(f"\n🎯 Training complete. Best Val F1: {best_f1:.4f}")

# ✅ 학습 실행
train_model(model, train_loader, val_loader, epochs=10, mc_samples=5)

