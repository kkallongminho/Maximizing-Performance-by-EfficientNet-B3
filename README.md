Maximizing-Performance-by-EfficientNet-B3

This repository contains experiments aimed at maximizing binary classification performance (real vs. fake images) using EfficientNet-B3. Each script applies a single enhancement technique (e.g., AutoAugment, MixUp, DropPath, Cosine LR, EMA, MC Dropout), making it easy to compare improvements against the baseline.

Original repo: kkallongminho/Maximizing-Performance-by-EfficientNet-B3

⸻

📁 Project Structure

Maximizing-Performance-by-EfficientNet-B3/
├─ BASELINE/
│  └─ baseline.py            # Baseline using torchvision EfficientNet-B3
├─ autoaugement/
│  └─ autoaugement.py        # AutoAugment (ImageNet policy)
├─ cosine/
│  └─ cosine.py              # Cosine Annealing learning rate scheduler
├─ droppath/
│  └─ droppath.py            # DropPath (Stochastic Depth)
├─ ema/
│  └─ ema.py                 # Exponential Moving Average (EMA)
├─ mcdropout/
│  └─ mcdropout.py           # Monte Carlo Dropout (uncertainty estimation)
├─ mixup/
│  └─ mixup.py               # MixUp data augmentation
└─ Result of the experiment.png  # Summary of experimental results

Each script applies one method only, making it straightforward to evaluate improvements over the baseline.

⸻

🧰 Requirements
	•	Python 3.8+
	•	PyTorch, torchvision
	•	timm (for tf_efficientnet_b3 in some scripts)
	•	scikit-learn (for F1-score)
	•	tqdm

Installation Example

pip install torch torchvision timm scikit-learn tqdm

GPU with CUDA is recommended for training.

⸻

🗂 Dataset Structure

All scripts assume an ImageFolder dataset structure:

Dataset/
├─ Train/
│  ├─ fake/  ...
│  └─ real/  ...
└─ Validation/
   ├─ fake/  ...
   └─ real/  ...

In each script, dataset paths are hardcoded (e.g., /kaggle/input/deepfake-and-real-images/Dataset/...).
You may:
	1.	Modify train_dir, val_dir, save_path variables directly, or
	2.	Create symbolic links matching the expected paths.

⸻

⚙️ Common Settings
	•	Input size: 300×300
	•	Preprocessing: Resize → ToTensor → Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
	•	Batch size: 32
	•	Optimizer: AdamW(lr=1e-4, weight_decay=1e-4) (may vary per script)
	•	Metrics: macro F1, accuracy
	•	Model saving: triggered when validation F1-score improves

Note: Scripts do not include random seed fixing. For reproducibility, set torch.manual_seed, numpy.random_seed, and cudnn.deterministic manually.

⸻

🚀 How to Run (Examples)

Each script is independent. Adjust dataset paths, then run:

1) Baseline

python BASELINE/baseline.py

	•	Backbone: torchvision.models.efficientnet_b3
	•	Classifier head: replaced with nn.Linear(in_features, 2)

2) AutoAugment

python autoaugement/autoaugement.py
