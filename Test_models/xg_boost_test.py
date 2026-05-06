import os, glob, csv
import numpy as np
import xgboost as xgb
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score

print("🚀 XGBOOST TEST SCRIPT STARTED")

# =========================
# CONFIG
# =========================
TEST_FOLDERS = [
    "Test folder directory.."
]


MODEL_PATH = "/media/corsair/New Volume/wirelessdataset/outputs/xgboost_36feat.json"

SAVE_DIR = "./outputs/test"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_FILE = os.path.join(SAVE_DIR, "xgboost_test_results.txt")
CSV_FILE    = os.path.join(SAVE_DIR, "xgboost_predictions.csv")

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

    # normalization (same as training)
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True)
    X = (X - mean) / (std + 1e-8)

    print("📥 Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    print("🚀 Running inference...")
    preds = model.predict(X)

    acc = accuracy_score(y, preds)

    print(f"\n🔥 XGBOOST TEST ACCURACY: {acc:.4f}")

    # =========================
    # SAVE RESULTS
    # =========================
    print("💾 Saving results...")

    with open(RESULT_FILE, "w") as f:
        f.write(f"XGBoost Test Accuracy: {acc:.6f}\n")
        f.write(f"Total Samples: {len(y)}\n")

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth", "prediction"])

        for gt, pred in zip(y, preds):
            writer.writerow([gt, pred])

    print(f"✅ Results saved to: {SAVE_DIR}")

# =========================
if __name__ == "__main__":
    test()
