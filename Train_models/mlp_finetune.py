import os, glob, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from scipy.stats import skew, kurtosis

print("🚀 Script started...")

# =========================
# CONFIG
# =========================
DATASET_ROOT = "./Datasets"
VERSIONS = [f"V{i}" for i in range(1, 21)]

BATCH_SIZE = 64
FINETUNE_EPOCHS = 100
LR = 1e-4   # 🔥 lower LR for finetuning
NUM_CLASSES = 16

EXP_NAME = "mlp_kl_div_36feat"
SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📌 Using device: {device}")

# pretrained model
PRETRAINED_PATH = os.path.join(SAVE_DIR, f"{EXP_NAME}_best.pt")

# finetune outputs
FT_BEST_PATH = os.path.join(SAVE_DIR, "mlp_kl_div_finetuned_best.pt")
FT_LAST_PATH = os.path.join(SAVE_DIR, "mlp_kl_div_finetuned_last.pt")
FT_LOG_PATH  = os.path.join(SAVE_DIR, "mlp_kl_div_finetune_log.csv")

# =========================
# HELPERS
# =========================
def find_file(folder, keyword):
    files = glob.glob(os.path.join(folder, f"*{keyword}*.mat"))
    if not files:
        raise FileNotFoundError(f"{keyword} not found in {folder}")
    return files[0]

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
    print(f"📂 Loading {folder}")

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
    print("🔄 Preparing data...")

    X_list, y_list = [], []

    for v in VERSIONS:
        X, y = load_version(v)
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print("📏 Normalizing...")
    mean = X.mean(0, keepdim=True)
    std  = X.std(0, keepdim=True)
    X = (X - mean) / (std + 1e-8)

    print("🔀 Splitting...")
    perm = torch.randperm(X.shape[0])
    split = int(0.8 * X.shape[0])

    return (
        X[perm[:split]].to(device),
        X[perm[split:]].to(device),
        y[perm[:split]].to(device),
        y[perm[split:]].to(device)
    )

# =========================
# MODEL
# =========================
class FNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# FINETUNE
# =========================
def finetune():
    print("🔥 Finetuning started...")

    X_train, X_val, y_train, y_val = prepare_data()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = FNN(X_train.shape[1]).to(device)

    print("📥 Loading pretrained model...")
    if not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(f"❌ Pretrained model not found at {PRETRAINED_PATH}")

    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0

    with open(FT_LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "train_acc", "val_acc"])

    for epoch in range(FINETUNE_EPOCHS):
        model.train()

        total_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            logits = model(xb)

            one_hot = F.one_hot(yb, NUM_CLASSES).float()
            targets = 0.9 * one_hot + 0.1 / NUM_CLASSES

            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), FT_BEST_PATH)

        torch.save(model.state_dict(), FT_LAST_PATH)

        with open(FT_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_loss, train_acc, val_acc])

        print(f"[FT] Epoch {epoch+1:03d} | Loss={avg_loss:.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}")

    print(f"\n🔥 Best Finetuned Accuracy: {best_acc:.4f}")

# =========================
if __name__ == "__main__":
    print("🚀 Entering main...")
    finetune()