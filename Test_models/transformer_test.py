import os, glob, csv
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.stats import skew, kurtosis

print("🚀 TRANSFORMER TEST SCRIPT STARTED")

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
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 IMPORTANT: FIX PATH
MODEL_PATH = "/media/corsair/New Volume/wirelessdataset/outputs/transformer_36feat_randomsplit_best.pt"

# SAVE DIR
SAVE_DIR = "./outputs/test"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_FILE = os.path.join(SAVE_DIR, "transformer_test_results.txt")
CSV_FILE    = os.path.join(SAVE_DIR, "transformer_predictions.csv")

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
class ChannelTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(TOKEN_DIM, D_MODEL)

        self.pos_embed = nn.Parameter(torch.randn(1, NUM_TOKENS, D_MODEL) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)

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

        self.classifier = nn.Sequential(
            nn.LayerNorm(D_MODEL),
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

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)

        return self.classifier(x[:, 0, :])

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

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # normalization (same style as training)
    mean = X.mean(0, keepdim=True)
    std  = X.std(0, keepdim=True)
    X = (X - mean) / (std + 1e-8)

    X = X.to(device)
    y = y.to(device)

    # model
    model = ChannelTransformer().to(device)

    print("📥 Loading transformer weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.eval()

    print("🚀 Running inference...")

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)

    acc = (preds == y).float().mean().item()

    print(f"\n🔥 TRANSFORMER TEST ACCURACY: {acc:.4f}")

    # =========================
    # SAVE LOGS
    # =========================
    print("💾 Saving results...")

    with open(RESULT_FILE, "w") as f:
        f.write(f"Transformer Test Accuracy: {acc:.6f}\n")
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
