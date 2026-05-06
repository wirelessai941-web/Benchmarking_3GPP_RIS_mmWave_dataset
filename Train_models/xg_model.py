import os, glob, csv
import numpy as np
import xgboost as xgb
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score

# =========================
# CONFIG
# =========================
DATASET_ROOT = "./Datasets"
VERSIONS = [f"V{i}" for i in range(1, 21)]

EXP_NAME = "xgboost_36feat"
SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

model_path = os.path.join(SAVE_DIR, f"{EXP_NAME}.json")
log_path   = os.path.join(SAVE_DIR, f"{EXP_NAME}_log.csv")

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

    skewness = skew(x, axis=1)
    kurt     = kurtosis(x, axis=1)

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
    print("▶ Loading ALL data...")

    X_list, y_list = [], []

    for v in VERSIONS:
        print(f"Loading {v}")
        X, y = load_version(v)
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    # Normalize (same as MLP)
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True)
    X = (X - mean) / (std + 1e-8)

    # Random split
    N = X.shape[0]
    perm = np.random.permutation(N)

    split = int(0.8 * N)
    return X[perm[:split]], X[perm[split:]], y[perm[:split]], y[perm[split:]]

# =========================
# TRAIN XGBOOST
# =========================
def train():
    X_train, X_val, y_train, y_val = prepare_data()

    print("\n▶ Training XGBoost...")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=16,
        tree_method="hist",
        eval_metric="mlogloss"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    # Predictions
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc   = accuracy_score(y_val, val_preds)

    print(f"\n🔥 Final Train Acc: {train_acc:.4f}")
    print(f"🔥 Final Val   Acc: {val_acc:.4f}")

    # Save model
    model.save_model(model_path)

    # Save log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_acc", "val_acc"])
        writer.writerow([train_acc, val_acc])

    print(f"\n📁 Saved:")
    print(f"Model → {model_path}")
    print(f"Log   → {log_path}")

# =========================
if __name__ == "__main__":
    train()