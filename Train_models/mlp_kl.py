# import os, glob, csv
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from scipy.io import loadmat

# # =========================
# # CONFIG
# # =========================
# DATASET_ROOT = "./Datasets"
# VERSIONS = [f"V{i}" for i in range(1, 21)]

# BATCH_SIZE = 64
# EPOCHS = 100
# LR = 5e-4
# NUM_CLASSES = 16

# EXP_NAME = "mlp_kl_div"
# SAVE_DIR = "./outputs"
# os.makedirs(SAVE_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =========================
# # FILE PATHS
# # =========================
# best_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_best.pt")
# last_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_last.pt")
# log_path        = os.path.join(SAVE_DIR, f"{EXP_NAME}_log.csv")

# # =========================
# # LOAD HELPERS
# # =========================
# def find_file(folder, keyword):
#     files = glob.glob(os.path.join(folder, f"*{keyword}*.mat"))
#     if not files:
#         raise FileNotFoundError(f"{keyword} not found in {folder}")
#     return files[0]

# def extract_features(H, G, H_D):
#     def stats(x):
#         x = np.abs(x)
#         return np.stack([
#             x.mean(axis=1),
#             x.std(axis=1),
#             x.max(axis=1),
#             x.min(axis=1)
#         ], axis=1)

#     return np.concatenate([
#         stats(H),
#         stats(G),
#         stats(H_D)
#     ], axis=1).astype(np.float32)

# def load_version(v):
#     folder = os.path.join(DATASET_ROOT, v)

#     H   = loadmat(find_file(folder, "BS_RIS_channel"))["H"]
#     G   = loadmat(find_file(folder, "RIS_UE_channel"))["G"]
#     H_D = loadmat(find_file(folder, "direct_channel"))["H_D"]

#     X = extract_features(H, G, H_D)

#     cqi_mat = loadmat(find_file(folder, "CQI_sample"))
#     key = [k for k in cqi_mat if not k.startswith("_")][0]
#     y = cqi_mat[key].flatten().astype(np.int64)

#     if y.min() == 1:
#         y -= 1

#     return X, y

# # =========================
# # DATA PREP
# # =========================
# def prepare_data():
#     print("▶ Loading ALL data...")

#     X_list, y_list = [], []

#     for v in VERSIONS:
#         print(f"Loading {v}")
#         X, y = load_version(v)
#         X_list.append(X)
#         y_list.append(y)

#     X = np.concatenate(X_list)
#     y = np.concatenate(y_list)

#     X = torch.tensor(X, dtype=torch.float32, device=device)
#     y = torch.tensor(y, dtype=torch.long, device=device)

#     # Normalize
#     mean = X.mean(dim=0, keepdim=True)
#     std  = X.std(dim=0, keepdim=True)
#     X = (X - mean) / (std + 1e-8)

#     # Random split
#     N = X.shape[0]
#     perm = torch.randperm(N, device=device)

#     split = int(0.8 * N)
#     train_idx = perm[:split]
#     val_idx   = perm[split:]

#     return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

# # =========================
# # MODEL
# # =========================
# class FNN(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(128, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(32, NUM_CLASSES)
#         )

#     def forward(self, x):
#         return self.net(x)

# # =========================
# # TRAIN
# # =========================
# def train():
#     X_train, X_val, y_train, y_val = prepare_data()

#     train_loader = DataLoader(
#         TensorDataset(X_train, y_train),
#         batch_size=BATCH_SIZE,
#         shuffle=True
#     )

#     val_loader = DataLoader(
#         TensorDataset(X_val, y_val),
#         batch_size=BATCH_SIZE
#     )

#     model = FNN(X_train.shape[1]).to(device)

#     # KL Divergence Loss
#     criterion = nn.KLDivLoss(reduction='batchmean')
#     optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

#     best_acc = 0

#     # Create log file
#     with open(log_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

#     for epoch in range(EPOCHS):
#         model.train()

#         total_loss = 0
#         correct_train = 0
#         total_train = 0

#         for xb, yb in train_loader:
#             outputs = model(xb)

#             log_probs = F.log_softmax(outputs, dim=1)
#             targets   = F.one_hot(yb, num_classes=NUM_CLASSES).float()

#             loss = criterion(log_probs, targets)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             preds = outputs.argmax(dim=1)
#             correct_train += (preds == yb).sum().item()
#             total_train += yb.size(0)

#         avg_loss = total_loss / len(train_loader)
#         train_acc = correct_train / total_train

#         # ---- VALIDATION ----
#         model.eval()
#         correct, total = 0, 0

#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 preds = model(xb).argmax(dim=1)
#                 correct += (preds == yb).sum().item()
#                 total += yb.size(0)

#         val_acc = correct / total

#         scheduler.step()

#         # Save best model
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), best_model_path)

#         # Save last model
#         torch.save(model.state_dict(), last_model_path)

#         # Log
#         with open(log_path, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch+1, avg_loss, train_acc, val_acc])

#         print(f"Epoch {epoch+1:02d} | Loss={avg_loss:.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}")

#     print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f}")
#     print(f"\n📁 Saved files:")
#     print(f"Best Model : {best_model_path}")
#     print(f"Last Model : {last_model_path}")
#     print(f"Logs       : {log_path}")

# # =========================
# if __name__ == "__main__":
#     train()
# import os, glob, csv
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from scipy.io import loadmat
# from scipy.stats import skew

# # =========================
# # CONFIG
# # =========================
# DATASET_ROOT = "./Datasets"
# VERSIONS = [f"V{i}" for i in range(1, 21)]

# BATCH_SIZE = 64
# EPOCHS = 60
# LR = 5e-4
# NUM_CLASSES = 16

# EXP_NAME = "mlp_kl_div_21feat"
# SAVE_DIR = "./outputs"
# os.makedirs(SAVE_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =========================
# # FILE PATHS
# # =========================
# best_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_best.pt")
# last_model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_last.pt")
# log_path        = os.path.join(SAVE_DIR, f"{EXP_NAME}_log.csv")

# # =========================
# # HELPERS
# # =========================
# def find_file(folder, keyword):
#     files = glob.glob(os.path.join(folder, f"*{keyword}*.mat"))
#     if not files:
#         raise FileNotFoundError(f"{keyword} not found in {folder}")
#     return files[0]

# def compute_stats(x):
#     x = np.abs(x)

#     mean   = x.mean(axis=1)
#     std    = x.std(axis=1)
#     maxv   = x.max(axis=1)
#     minv   = x.min(axis=1)
#     median = np.median(x, axis=1)
#     energy = (x**2).mean(axis=1)
#     skewness = skew(x, axis=1)

#     return np.stack([
#         mean, std, maxv, minv,
#         median, energy, skewness
#     ], axis=1)

# def extract_features(H, G, H_D):
#     return np.concatenate([
#         compute_stats(H),
#         compute_stats(G),
#         compute_stats(H_D)
#     ], axis=1).astype(np.float32)

# def load_version(v):
#     folder = os.path.join(DATASET_ROOT, v)

#     H   = loadmat(find_file(folder, "BS_RIS_channel"))["H"]
#     G   = loadmat(find_file(folder, "RIS_UE_channel"))["G"]
#     H_D = loadmat(find_file(folder, "direct_channel"))["H_D"]

#     X = extract_features(H, G, H_D)

#     cqi_mat = loadmat(find_file(folder, "CQI_sample"))
#     key = [k for k in cqi_mat if not k.startswith("_")][0]
#     y = cqi_mat[key].flatten().astype(np.int64)

#     if y.min() == 1:
#         y -= 1

#     return X, y

# # =========================
# # DATA PREP
# # =========================
# def prepare_data():
#     print("▶ Loading ALL data...")

#     X_list, y_list = [], []

#     for v in VERSIONS:
#         print(f"Loading {v}")
#         X, y = load_version(v)
#         X_list.append(X)
#         y_list.append(y)

#     X = np.concatenate(X_list)
#     y = np.concatenate(y_list)

#     X = torch.tensor(X, dtype=torch.float32, device=device)
#     y = torch.tensor(y, dtype=torch.long, device=device)

#     # Normalize
#     mean = X.mean(dim=0, keepdim=True)
#     std  = X.std(dim=0, keepdim=True)
#     X = (X - mean) / (std + 1e-8)

#     # Random split
#     N = X.shape[0]
#     perm = torch.randperm(N, device=device)

#     split = int(0.8 * N)
#     return X[perm[:split]], X[perm[split:]], y[perm[:split]], y[perm[split:]]

# # =========================
# # MODEL (slightly wider)
# # =========================
# class FNN(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(128, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),

#             nn.Linear(64, NUM_CLASSES)
#         )

#     def forward(self, x):
#         return self.net(x)

# # =========================
# # TRAIN
# # =========================
# def train():
#     X_train, X_val, y_train, y_val = prepare_data()

#     train_loader = DataLoader(
#         TensorDataset(X_train, y_train),
#         batch_size=BATCH_SIZE,
#         shuffle=True
#     )

#     val_loader = DataLoader(
#         TensorDataset(X_val, y_val),
#         batch_size=BATCH_SIZE
#     )

#     model = FNN(X_train.shape[1]).to(device)

#     criterion = nn.KLDivLoss(reduction='batchmean')
#     optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

#     best_acc = 0

#     # log file
#     with open(log_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

#     for epoch in range(EPOCHS):
#         model.train()

#         total_loss = 0
#         correct_train = 0
#         total_train = 0

#         for xb, yb in train_loader:
#             outputs = model(xb)

#             log_probs = F.log_softmax(outputs, dim=1)
#             targets   = F.one_hot(yb, NUM_CLASSES).float()

#             loss = criterion(log_probs, targets)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             preds = outputs.argmax(dim=1)
#             correct_train += (preds == yb).sum().item()
#             total_train += yb.size(0)

#         avg_loss = total_loss / len(train_loader)
#         train_acc = correct_train / total_train

#         # validation
#         model.eval()
#         correct, total = 0, 0

#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 preds = model(xb).argmax(dim=1)
#                 correct += (preds == yb).sum().item()
#                 total += yb.size(0)

#         val_acc = correct / total
#         scheduler.step()

#         # save best
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), best_model_path)

#         torch.save(model.state_dict(), last_model_path)

#         with open(log_path, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch+1, avg_loss, train_acc, val_acc])

#         print(f"Epoch {epoch+1:02d} | Loss={avg_loss:.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}")

#     print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f}")
#     print(f"📁 Saved to: {SAVE_DIR}")

# # =========================
# if __name__ == "__main__":
#     train()
import os, glob, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from scipy.stats import skew, kurtosis

# =========================
# CONFIG
# =========================
DATASET_ROOT = "./Datasets"
VERSIONS = [f"V{i}" for i in range(1, 21)]

BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4
NUM_CLASSES = 16

EXP_NAME = "mlp_kl_div_36feat"
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

    mean   = x.mean(axis=1)
    std    = x.std(axis=1)
    maxv   = x.max(axis=1)
    minv   = x.min(axis=1)
    median = np.median(x, axis=1)
    energy = (x**2).mean(axis=1)

    skewness  = skew(x, axis=1)
    kurt      = kurtosis(x, axis=1)

    p25 = np.percentile(x, 25, axis=1)
    p75 = np.percentile(x, 75, axis=1)

    range_ = maxv - minv
    log_energy = np.log(energy + 1e-8)

    return np.stack([
        mean, std, maxv, minv,
        median, energy,
        skewness, kurt,
        p25, p75,
        range_, log_energy
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

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    # normalization
    mean = X.mean(0, keepdim=True)
    std  = X.std(0, keepdim=True)
    X = (X - mean) / (std + 1e-8)

    # split
    N = X.shape[0]
    perm = torch.randperm(N, device=device)

    split = int(0.8 * N)
    return X[perm[:split]], X[perm[split:]], y[perm[:split]], y[perm[split:]]

# =========================
# MODEL (WIDER)
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
# TRAIN
# =========================
def train():
    X_train, X_val, y_train, y_val = prepare_data()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = FNN(X_train.shape[1]).to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    best_acc = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_acc", "val_acc"])

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            logits = model(xb)

            # label smoothing
            one_hot = F.one_hot(yb, NUM_CLASSES).float()
            eps = 0.1
            targets = (1 - eps) * one_hot + eps / NUM_CLASSES

            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
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
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        torch.save(model.state_dict(), last_model_path)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_loss, train_acc, val_acc])

        print(f"Epoch {epoch+1:02d} | Loss={avg_loss:.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}")

    print(f"\n🔥 Best Val Accuracy: {best_acc:.4f}")

# =========================
if __name__ == "__main__":
    train()