# Pitch Type Classification
# PART 3: Modeling


# Baseline: Logistic Regression
# Main Model: XGBoost



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import time


# 1. LOAD PREPROCESSED DATA
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")
classes = np.load("label_classes.npy", allow_pickle=True)

PITCH_NAMES = {
    'FF': '4-Seam FB', 'SI': 'Sinker',   'FC': 'Cutter',
    'SL': 'Slider',    'ST': 'Swp. Curve','CU': 'Curveball',
    'CH': 'Changeup',  'FS': 'Splitter',  'KC': 'Knkl. Curve',
    'SV': 'SV',        'FA': 'FA',        'FO': 'Forkball',
    'EP': 'Eephus',    'KN': 'Knuckleball'
}
display_names = [PITCH_NAMES.get(c, c) for c in classes]

print(f"Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
print(f"Classes ({len(classes)}): {list(classes)}\n")



# 2. BASELINE — LOGISTIC REGRESSION
print("=" * 50)
print("BASELINE: Logistic Regression")
print("=" * 50)

t0 = time.time()
lr = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)
lr_time = time.time() - t0

lr_preds = lr.predict(X_test)
lr_acc   = accuracy_score(y_test, lr_preds)

print(f"Training time: {lr_time:.1f}s")
print(f"Test Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)\n")
print("Per-class Report:")
print(classification_report(y_test, lr_preds, target_names=display_names, digits=3))



# 3. MAIN MODEL — XGBOOST
print("=" * 50)
print("MAIN MODEL: XGBoost")
print("=" * 50)

t0 = time.time()
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50               # Print loss every 50 rounds
)
xgb_time = time.time() - t0

xgb_preds = xgb.predict(X_test)
xgb_acc   = accuracy_score(y_test, xgb_preds)

print(f"\nTraining time: {xgb_time:.1f}s")
print(f"Test Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)\n")
print("Per-class Report:")
print(classification_report(y_test, xgb_preds, target_names=display_names, digits=3))


# 4. MODEL COMPARISON SUMMARY
print("=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"{'Model':<25} {'Accuracy':>10} {'Train Time':>12}")
print("-" * 50)
print(f"{'Logistic Regression':<25} {lr_acc*100:>9.2f}% {lr_time:>10.1f}s")
print(f"{'XGBoost':<25} {xgb_acc*100:>9.2f}% {xgb_time:>10.1f}s")



# 5. CONFUSION MATRIX — XGBOOST
fig, ax = plt.subplots(figsize=(13, 10))

cm = confusion_matrix(y_test, xgb_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='.2f')

ax.set_title('XGBoost — Normalized Confusion Matrix\n(Row = True Label, Value = Recall per Class)',
             fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("model_confusion_matrix.png", dpi=150)
plt.show()
print("\nSaved: model_confusion_matrix.png")


# 6. FEATURE IMPORTANCE
FEATURE_NAMES = [
    'release_speed', 'release_spin_rate',
    'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'plate_x', 'plate_z',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az'
]

importances = xgb.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(FEATURE_NAMES)))

bars = ax.bar(
    range(len(FEATURE_NAMES)),
    importances[sorted_idx],
    color=colors
)
ax.set_xticks(range(len(FEATURE_NAMES)))
ax.set_xticklabels([FEATURE_NAMES[i] for i in sorted_idx], rotation=45, ha='right')
ax.set_ylabel('Feature Importance (XGBoost Gain)', fontsize=11)
ax.set_title('Feature Importance — XGBoost Pitch Type Classifier', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("model_feature_importance.png", dpi=150)
plt.show()
print("Saved: model_feature_importance.png")


# 7. DEEP LEARNING — MLP (EXTENSION)
# A pitch is represented as a feature vector here, same
# as logistic regression and XGBoost. This lets us do a clean
# 3 way comparison and assess whether deep learning
# adds value over gradient boosting on this problem.

print("=" * 50)
print("DEEP LEARNING: MLP (Feedforward Neural Network)")
print("=" * 50)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Convert to tensors 
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=512, shuffle=True
)

n_features = X_train.shape[1]
n_classes  = len(classes)

# ARCHITECTURE 
# Three hidden layers with BatchNorm and Dropout.
# Dropout rate is (0.3) — enough to regularize without
# crippling a model that already has limited input dimensionality.
class PitchMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PitchMLP(n_features, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler — reduces LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Training loop 
EPOCHS = 40
train_losses, val_losses = [], []

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)

    avg_train_loss = epoch_loss / len(X_train_t)

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_logits = model(X_test_t.to(device))
        val_loss   = criterion(val_logits, y_test_t.to(device)).item()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3}/{EPOCHS}  |  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

mlp_time = time.time() - t0

# Evaluation
model.eval()
with torch.no_grad():
    mlp_preds = model(X_test_t.to(device)).argmax(dim=1).cpu().numpy()

mlp_acc = accuracy_score(y_test, mlp_preds)
print(f"\nTraining time: {mlp_time:.1f}s")
print(f"Test Accuracy: {mlp_acc:.4f} ({mlp_acc*100:.2f}%)\n")
print("Per-class Report:")
print(classification_report(y_test, mlp_preds, target_names=display_names, digits=3))

# Training curve 
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", linewidth=2)
ax.plot(range(1, EPOCHS + 1), val_losses,   label="Val Loss",   linewidth=2, linestyle="--")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
ax.set_title("MLP Training Curve — Pitch Type Classification", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("model_mlp_training_curve.png", dpi=150)
plt.show()
print("Saved: model_mlp_training_curve.png")



# 8. FINAL 3-WAY COMPARISON SUMMARY
print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"{'Model':<28} {'Accuracy':>10} {'Train Time':>12}")
print("-" * 55)
print(f"{'Logistic Regression':<28} {lr_acc*100:>9.2f}% {lr_time:>10.1f}s")
print(f"{'XGBoost':<28} {xgb_acc*100:>9.2f}% {xgb_time:>10.1f}s")
print(f"{'MLP (Neural Network)':<28} {mlp_acc*100:>9.2f}% {mlp_time:>10.1f}s")



# 9. SAVE ALL RESULTS
np.save("xgb_preds.npy", xgb_preds)
np.save("lr_preds.npy",  lr_preds)
np.save("mlp_preds.npy", mlp_preds)

