import os, glob, csv
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

# Mamba
from mamba_ssm.modules.mamba_simple import Mamba

print("🚀 MAMBA TEST SCRIPT STARTED")

# =========================
# CONFIG
# =========================
TEST_FOLDERS = [
    "Test folder directory.."
]

NUM_CLASSES = 16
NUM_TOKENS = 3
TOKEN_DIM = 12
D_MODEL = 64
NUM_LAYERS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 FIX PATH
MODEL_PATH = "/media/corsair/New Volume/wirelessdataset/outputs/mamba_final_fixed_best.pt"

# SAVE DIR
SAVE_DIR = "./outputs/test"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_FILE = os.path.join(SAVE_DIR, "mamba_test_results.txt")
CSV_FILE    = os.path.join(SAVE_DIR, "mamba_predictions.csv")

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

def load_test_folder(folder):
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
# TEST
# =========================
def test():
    print("🔥 Loading test data...")

    X_list, y_list = [], []

    for folder in TEST_FOLDERS:
        X, y = load_test_folder(folder)
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    print("📊 Total test samples:", X.shape[0])

    # 🔥 SAME normalization as training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    # model
    model = ChannelMamba().to(device)

    print("📥 Loading Mamba weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.eval()

    print("🚀 Running inference...")

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)

    acc = (preds == y).float().mean().item()

    print(f"\n🔥 MAMBA TEST ACCURACY: {acc:.4f}")

    # =========================
    # SAVE RESULTS
    # =========================
    print("💾 Saving results...")

    with open(RESULT_FILE, "w") as f:
        f.write(f"Mamba Test Accuracy: {acc:.6f}\n")
        f.write(f"Total Samples: {len(y)}\n")

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth", "prediction"])

        for gt, pred in zip(y.cpu().numpy(), preds.cpu().numpy()):
            writer.writerow([gt, pred])

    print(f"✅ Saved results to: {SAVE_DIR}")

# =========================
if __name__ == "__main__":
    test()
