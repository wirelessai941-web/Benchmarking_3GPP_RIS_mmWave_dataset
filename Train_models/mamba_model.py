import os, glob, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

# =========================
# MAMBA (CPU SAFE)
# =========================
from mamba_ssm.modules.mamba_simple import Mamba

# =========================
# CONFIG
# =========================
DATASET_ROOT = "./Datasets"
VERSIONS = [f"V{i}" for i in range(1, 21)]

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4   # 🔥 Stable LR for Mamba
NUM_CLASSES = 16

NUM_TOKENS = 3
TOKEN_DIM = 12
D_MODEL = 64
NUM_LAYERS = 3

EXP_NAME = "mamba_final_fixed"
SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_best.pt")
last_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_last.pt")
log_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_log.csv")

# =========================
# HELPERS
# =========================
def find_file(folder, keyword):
    return glob.glob(os.path.join(folder, f"*{keyword}*.mat"))[0]

def compute_stats(x):
    x = np.abs(x)
    return np.stack([
        x.mean(axis=1), x.std(axis=1), x.max(axis=1), x.min(axis=1),
        np.median(x, axis=1), (x**2).mean(axis=1),
        skew(x, axis=1), kurtosis(x, axis=1),
        np.percentile(x, 25, axis=1), np.percentile(x, 75, axis=1),
        x.max(axis=1) - x.min(axis=1),
        np.log((x**2).mean(axis=1) + 1e-8)
    ], axis=1)

def extract_features(H, G, H_D):
    return np.concatenate([
        compute_stats(H),
        compute_stats(G),
        compute_stats(H_D)
    ], axis=1).astype(np.float32)

def load_version(v):
    folder = os.path.join(DATASET_ROOT, v)

    H   = loadmat(find_file(folder, "BS_RIS_channel"))["H"]
    G   = loadmat(find_file(folder, "RIS_UE_channel"))["G"]
    H_D = loadmat(find_file(folder, "direct_channel"))["H_D"]

    X = extract_features(H, G, H_D)

    cqi_mat = loadmat(find_file(folder, "CQI_sample"))
    key = [k for k in cqi_mat if not k.startswith("_")][0]
    y = cqi_mat[key].flatten().astype(np.int64)

    if y.min() == 1:
        y -= 1

    return X, y

# =========================
# DATA PREP
# =========================
def prepare_data():
    X_list, y_list = [], []

    for v in VERSIONS:
        print(f"Loading {v}")
        X, y = load_version(v)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    perm = np.random.permutation(len(X_all))
    split = int(0.8 * len(X_all))

    return (
        torch.tensor(X_all[perm[:split]], dtype=torch.float32),
        torch.tensor(X_all[perm[split:]], dtype=torch.float32),
        torch.tensor(y_all[perm[:split]], dtype=torch.long),
        torch.tensor(y_all[perm[split:]], dtype=torch.long)
    )

# =========================
# MODEL
# =========================
class ChannelMamba(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(TOKEN_DIM, D_MODEL)

        self.pos_embed = nn.Parameter(
            torch.randn(1, NUM_TOKENS, D_MODEL) * 0.02
        )

        self.layers = nn.ModuleList([
            Mamba(d_model=D_MODEL, d_state=16, d_conv=4, expand=2)
            for _ in range(NUM_LAYERS)
        ])

        self.norm = nn.LayerNorm(D_MODEL)

        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES)
        )

        # 🔥 FIXED INIT
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:   # ✅ FIX
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)

        x = x.view(B, NUM_TOKENS, TOKEN_DIM)
        x = self.input_proj(x)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.classifier(x)

# =========================
# TRAIN
# =========================
def train():
    X_train, X_val, y_train, y_val = prepare_data()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = ChannelMamba().to(device)

    print("\n▶ Stable Mamba Training Started\n")

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.7)

    best_acc = 0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "train_acc", "val_acc"])

    for epoch in range(EPOCHS):
        model.train()

        total_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)

            # 🔥 FIX: Clamp logits
            logits = torch.clamp(logits, -10, 10)

            one_hot = F.one_hot(yb, NUM_CLASSES).float()
            targets = 0.9 * one_hot + 0.1 / NUM_CLASSES

            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

            if torch.isnan(loss):
                print("⚠️ NaN detected — skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()

            # 🔥 FIX: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        avg_loss  = total_loss / len(train_loader)

        # VALIDATION
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        torch.save(model.state_dict(), last_model_path)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_loss, train_acc, val_acc])

        print(f"Epoch {epoch+1:03d} | Loss={avg_loss:.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}")

    print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f}")

# =========================
if __name__ == "__main__":
    train()