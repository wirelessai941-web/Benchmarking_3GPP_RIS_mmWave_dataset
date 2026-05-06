import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

print("🚀 DISTANCE MATRIX SCRIPT STARTED")

# =========================
# CONFIG
# =========================
TRAIN_ROOT = "./Datasets"
TRAIN_VERSIONS = [f"V{i}" for i in range(1, 21)]

TEST_FOLDERS = [
    "/media/corsair/New Volume/test_Datasets/V21",
    "/media/corsair/New Volume/test_Datasets/V22",
    "/media/corsair/New Volume/test_Datasets/V23"
]

SAVE_DIR = "./outputs/distance_matrix"
os.makedirs(SAVE_DIR, exist_ok=True)

labels = ["V21", "V22", "V23"]

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

def load_folder(folder):
    H   = loadmat(find_file(folder, "BS_RIS_channel"))["H"]
    G   = loadmat(find_file(folder, "RIS_UE_channel"))["G"]
    H_D = loadmat(find_file(folder, "direct_channel"))["H_D"]
    return extract_features(H, G, H_D)

# =========================
# LOAD TRAIN DATA
# =========================
print("📥 Loading TRAIN data...")
X_train = np.concatenate([
    load_folder(os.path.join(TRAIN_ROOT, v))
    for v in TRAIN_VERSIONS
])

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# =========================
# DISTANCE FUNCTIONS
# =========================
def compute_kl(X_train, X_test):
    vals = []
    for i in range(X_train.shape[1]):
        p = X_train[:, i]
        q = X_test[:, i]

        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())

        p_hist, bins = np.histogram(p, bins=50, range=(min_val, max_val))
        q_hist, _    = np.histogram(q, bins=bins)

        p_hist = p_hist / (p_hist.sum() + 1e-12)
        q_hist = q_hist / (q_hist.sum() + 1e-12)

        eps = 1e-10
        p_hist = np.clip(p_hist, eps, None)
        q_hist = np.clip(q_hist, eps, None)

        kl_pq = np.sum(p_hist * np.log(p_hist / q_hist))
        kl_qp = np.sum(q_hist * np.log(q_hist / p_hist))

        vals.append(0.5 * (kl_pq + kl_qp))

    return np.mean(vals)

def compute_wass(X_train, X_test):
    return np.mean([
        wasserstein_distance(X_train[:, i], X_test[:, i])
        for i in range(X_train.shape[1])
    ])

def compute_mmd(X, Y):
    XX = rbf_kernel(X, X, gamma=0.5)
    YY = rbf_kernel(Y, Y, gamma=0.5)
    XY = rbf_kernel(X, Y, gamma=0.5)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# =========================
# BUILD MATRIX
# =========================
matrix = np.zeros((3, 3))  # rows=metrics, cols=test sets

for j, folder in enumerate(TEST_FOLDERS):
    print(f"\n📂 Processing {labels[j]}")

    X_test = load_folder(folder)
    X_test = scaler.transform(X_test)

    matrix[0, j] = compute_kl(X_train, X_test)
    matrix[1, j] = compute_wass(X_train, X_test)

    idx1 = np.random.choice(len(X_train), 2000, replace=False)
    idx2 = np.random.choice(len(X_test), 2000, replace=False)
    matrix[2, j] = compute_mmd(X_train[idx1], X_test[idx2])

# =========================
# PRINT MATRIX
# =========================
print("\n🔥 Distance Matrix:")
print(matrix)

# Save raw matrix
np.savetxt(os.path.join(SAVE_DIR, "distance_matrix.txt"), matrix)

# =========================
# PLOT MATRIX WITH VALUES
# =========================
plt.figure(figsize=(6, 5))

im = plt.imshow(matrix, cmap="viridis")
plt.colorbar(im)

x_labels = ["V21", "V22", "V23"]
y_labels = ["KL", "Wasserstein", "MMD"]

plt.xticks(range(3), x_labels)
plt.yticks(range(3), y_labels)

# Annotate values
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = matrix[i, j]
        plt.text(j, i, f"{value:.3f}",
                 ha="center", va="center",
                 color="white" if value > matrix.max()/2 else "black",
                 fontsize=11, fontweight="bold")

plt.title("Distance Matrix (Train vs Test Sets)")
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "distance_matrix_annotated.png"), dpi=300)

print("\n✅ Saved matrix image + values in:", SAVE_DIR)