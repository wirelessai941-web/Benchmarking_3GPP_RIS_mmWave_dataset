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
# CONFIG
# =========================
DATASET_ROOT = "./Datasets"
VERSIONS     = [f"V{i}" for i in range(1, 21)]

BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 5e-4
NUM_CLASSES = 16

# Transformer config
NUM_TOKENS = 3    # one per channel: H, G, H_D
TOKEN_DIM  = 12   # stats per channel
D_MODEL    = 64   # project each token to this dim
NHEAD      = 4    # attention heads (D_MODEL must be divisible by NHEAD)
NUM_LAYERS = 3    # transformer encoder layers
DIM_FF     = 128  # feedforward dim inside transformer

EXP_NAME = "transformer_36feat_randomsplit"
SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_best.pt")
last_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_last.pt")
log_path        = os.path.join(SAVE_DIR, f"{EXP_NAME}_log.csv")

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

    mean       = x.mean(axis=1)
    std        = x.std(axis=1)
    maxv       = x.max(axis=1)
    minv       = x.min(axis=1)
    median     = np.median(x, axis=1)
    energy     = (x**2).mean(axis=1)
    skewness   = skew(x, axis=1)
    kurt       = kurtosis(x, axis=1)
    p25        = np.percentile(x, 25, axis=1)
    p75        = np.percentile(x, 75, axis=1)
    range_     = maxv - minv
    log_energy = np.log(energy + 1e-8)

    return np.stack([
        mean, std, maxv, minv,
        median, energy,
        skewness, kurt,
        p25, p75,
        range_, log_energy
    ], axis=1)   # (N, 12)

def extract_features(H, G, H_D):
    return np.concatenate([
        compute_stats(H),    # (N, 12) → token 1
        compute_stats(G),    # (N, 12) → token 2
        compute_stats(H_D)   # (N, 12) → token 3
    ], axis=1).astype(np.float32)   # (N, 36)

def load_version(v):
    folder = os.path.join(DATASET_ROOT, v)

    H   = loadmat(find_file(folder, "BS_RIS_channel"))["H"]
    G   = loadmat(find_file(folder, "RIS_UE_channel"))["G"]
    H_D = loadmat(find_file(folder, "direct_channel"))["H_D"]

    X = extract_features(H, G, H_D)

    cqi_mat = loadmat(find_file(folder, "CQI_sample"))
    key = [k for k in cqi_mat if not k.startswith("_")][0]
    y   = cqi_mat[key].flatten().astype(np.int64)

    if y.min() == 1:
        y -= 1

    return X, y

# =========================
# DATA PREP (RANDOM 80/20)
# =========================
def prepare_data():
    print("▶ Split mode: RANDOM 80/20 across all 20 versions\n")

    X_list, y_list = [], []

    for v in VERSIONS:
        print(f"   Loading {v}...", flush=True)
        X, y = load_version(v)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)

    print(f"\n▶ Total data: {X_all.shape}")

    # Fit scaler on all data
    scaler = StandardScaler()
    X_all  = scaler.fit_transform(X_all)

    # Random 80/20 split
    N     = X_all.shape[0]
    perm  = np.random.permutation(N)
    split = int(0.8 * N)

    X_tr  = X_all[perm[:split]];  y_tr  = y_all[perm[:split]]
    X_val = X_all[perm[split:]];  y_val = y_all[perm[split:]]

    print(f"▶ Train: {X_tr.shape}  Val: {X_val.shape}")
    print(f"▶ Train label range: [{y_tr.min()}, {y_tr.max()}]")
    print(f"▶ Val   label range: [{y_val.min()}, {y_val.max()}]")

    print("\n▶ Train class distribution:")
    for i in range(NUM_CLASSES):
        count = (y_tr == i).sum()
        bar = "█" * (count // max(1, len(y_tr) // (NUM_CLASSES * 20)))
        print(f"   CQI {i:02d}: {count:6d}  {bar}")

    print("\n▶ Val class distribution:")
    for i in range(NUM_CLASSES):
        count = (y_val == i).sum()
        bar = "█" * (count // max(1, len(y_val) // (NUM_CLASSES * 20)))
        print(f"   CQI {i:02d}: {count:6d}  {bar}")

    X_tr  = torch.tensor(X_tr,  dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_tr  = torch.tensor(y_tr,  dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    return X_tr, X_val, y_tr, y_val

# =========================
# TRANSFORMER MODEL
# =========================
class ChannelTransformer(nn.Module):
    """
    Input:   (batch, 36) flat features
    Reshape: (batch, 3, 12) → 3 tokens: H, G, H_D

    Pipeline:
      1. Linear projection: 12 → D_MODEL per token
      2. Add learnable positional embedding
      3. Prepend CLS token
      4. Transformer encoder (attention across H, G, H_D tokens)
      5. CLS token → classifier head
    """
    def __init__(self):
        super().__init__()

        # Project each token from TOKEN_DIM → D_MODEL
        self.input_proj = nn.Linear(TOKEN_DIM, D_MODEL)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, NUM_TOKENS, D_MODEL) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, D_MODEL) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FF,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        # Classifier on CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        B = x.size(0)

        # (B, 36) → (B, 3, 12)
        x = x.view(B, NUM_TOKENS, TOKEN_DIM)

        # Project: (B, 3, 12) → (B, 3, D_MODEL)
        x = self.input_proj(x)

        # Add positional embedding
        x = x + self.pos_embed

        # Prepend CLS: (B, 3, D_MODEL) → (B, 4, D_MODEL)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # Transformer encoder
        x = self.transformer(x)

        # CLS token output → classify
        return self.classifier(x[:, 0, :])

# =========================
# TRAIN
# =========================
def train():
    X_train, X_val, y_train, y_val = prepare_data()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        pin_memory=(device.type == "cuda"),
        num_workers=0
    )

    model        = ChannelTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n▶ Model     : ChannelTransformer")
    print(f"▶ Tokens    : {NUM_TOKENS} (H, G, H_D) each {TOKEN_DIM}-dim → {D_MODEL}-dim")
    print(f"▶ Attn heads: {NHEAD}   Layers: {NUM_LAYERS}   FF dim: {DIM_FF}")
    print(f"▶ Params    : {total_params:,}")
    print(f"▶ Device    : {device}\n")

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.7
    )

    best_acc = 0.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "train_acc", "val_acc"])

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            xb, yb  = xb.to(device), yb.to(device)
            logits   = model(xb)

            # Label smoothing
            one_hot = F.one_hot(yb, NUM_CLASSES).float()
            eps     = 0.1
            targets = (1 - eps) * one_hot + eps / NUM_CLASSES

            log_probs = F.log_softmax(logits, dim=1)
            loss      = criterion(log_probs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += yb.size(0)

        train_acc = correct / total
        avg_loss  = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += yb.size(0)

        val_acc = correct / total
        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        torch.save(model.state_dict(), last_model_path)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_loss, train_acc, val_acc])

        marker = "  ◀ best" if is_best else ""
        print(f"Epoch {epoch+1:03d} | Loss={avg_loss:.4f} | "
              f"Train={train_acc:.4f} | Val={val_acc:.4f}{marker}")

    print(f"\n✅ Best Val Accuracy: {best_acc:.4f}")
    print(f"📁 Log  → {log_path}")
    print(f"📁 Best → {best_model_path}")

# =========================
if __name__ == "__main__":
    train()