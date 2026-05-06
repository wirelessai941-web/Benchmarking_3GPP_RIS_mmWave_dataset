import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.io

# ─────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────
def get_mat_data(file_path):
    """Robustly load data from a .mat file (v7.3 or older)."""
    try:
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if key not in ['#refs#', '#subsystem#']:
                    data = np.array(f[key])
                    if data.ndim > 1:
                        return data.T   # MATLAB v7.3 stores arrays transposed
                    return data
    except Exception:
        try:
            data = scipy.io.loadmat(file_path)
            for key in data.keys():
                if not key.startswith('__'):
                    return data[key]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return None


# ─────────────────────────────────────────────────────────────────
# Rich Eigenvalue Analysis Plot (4-panel)
# ─────────────────────────────────────────────────────────────────
def plot_eigenvalue_analysis(eigenvalues, folder_name, output_dir, n_top=6):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Eigenvalue Analysis - {folder_name}', fontsize=14)

    total_variance = np.sum(eigenvalues)
    explained_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_ratio)
    components = range(1, len(eigenvalues) + 1)

    # --- Plot 1: Log-Scale Scree ---
    ax = axes[0, 0]
    ax.plot(components, eigenvalues, 'o-', markersize=4, linewidth=1.5, color='steelblue')
    ax.axhline(y=1, color='red', linestyle='--', label='Kaiser (y=1)')
    ax.axvline(x=n_top, color='green', linestyle=':', label=f'Top {n_top}')
    ax.set_yscale('log')
    ax.set_title('Scree Plot (Log Scale)')
    ax.set_xlabel('Component')
    ax.set_ylabel('Eigenvalue (log)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)

    # --- Plot 2: Cumulative Explained Variance ---
    ax = axes[0, 1]
    ax.plot(components, cumulative_variance * 100, '-', color='darkorange', linewidth=2)
    ax.axhline(y=95, color='red',    linestyle='--', alpha=0.7, label='95% threshold')
    ax.axhline(y=99, color='purple', linestyle='--', alpha=0.7, label='99% threshold')
    ax.axvline(x=n_top, color='green', linestyle=':', label=f'Top {n_top}')
    ax.fill_between(list(components), cumulative_variance * 100, alpha=0.15, color='darkorange')
    ax.set_title('Cumulative Explained Variance (%)')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_ylim([0, 101])
    ax.legend()
    ax.grid(True, alpha=0.4)

    # --- Plot 3: Individual % Variance (Top 50, log y) ---
    ax = axes[1, 0]
    ax.bar(list(components[:50]), explained_ratio[:50] * 100,
           color='salmon', alpha=0.8, width=0.8)
    ax.axvline(x=n_top, color='green', linestyle=':', label=f'Top {n_top}')
    ax.set_title('Individual Explained Variance % (Top 50 Components)')
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)

    # --- Plot 4: Consecutive Eigenvalue Ratio (elbow detector) ---
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-12)
    ax = axes[1, 1]
    ax.plot(list(range(1, len(ratios) + 1)), ratios, 's-',
            color='mediumpurple', markersize=4, linewidth=1.5)
    ax.axvline(x=n_top, color='green', linestyle=':', label=f'Top {n_top}')
    ax.set_yscale('log')
    ax.set_title('Consecutive Eigenvalue Ratio (λᵢ / λᵢ₊₁)')
    ax.set_xlabel('Component')
    ax.set_ylabel('Ratio (log scale)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{folder_name}_eigenvalue_analysis.png'), dpi=150)
    plt.close()

    # Print variance summary
    for thresh in [0.90, 0.95, 0.99]:
        n_needed = np.searchsorted(cumulative_variance, thresh) + 1
        print(f"    Components for {thresh*100:.0f}% variance: {n_needed}")

    return explained_ratio   # Return for use in the summary table


# ─────────────────────────────────────────────────────────────────
# Per-Folder Analysis
# ─────────────────────────────────────────────────────────────────
def analyze_folder(folder_path, output_dir, n_top=6):
    folder_name = os.path.basename(folder_path)
    print(f"Processing {folder_name}...")

    files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    bs_ris_file = [f for f in files if 'BS_RIS_channel' in f]
    ris_ue_file = [f for f in files if 'RIS_UE_channel' in f]
    direct_file = [f for f in files if 'direct_channel' in f]

    if not (bs_ris_file and ris_ue_file and direct_file):
        print(f"  Missing channel files in {folder_name}")
        return None

    print("  Loading channels...")
    d_bs_ris = get_mat_data(os.path.join(folder_path, bs_ris_file[0]))
    d_ris_ue = get_mat_data(os.path.join(folder_path, ris_ue_file[0]))
    d_direct  = get_mat_data(os.path.join(folder_path, direct_file[0]))

    if d_bs_ris is None or d_ris_ue is None or d_direct is None:
        print(f"  Failed to load data for {folder_name}")
        return None

    # Flatten & Concatenate
    print("  Flattening and concatenating...")
    num_samples = d_bs_ris.shape[0]
    X = np.hstack([
        d_bs_ris.reshape(num_samples, -1),
        d_ris_ue.reshape(num_samples, -1),
        d_direct.reshape(num_samples, -1)
    ])

    if np.iscomplexobj(X):
        print("  Complex data → using magnitude...")
        X = np.abs(X)

    # Normalize before PCA — makes eigenvalues meaningful and Kaiser criterion valid
    print("  Normalizing (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA on normalized data
    print(f"  PCA on {X_scaled.shape} matrix...")
    n_comp = min(X_scaled.shape[0], X_scaled.shape[1], 100)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    eigenvalues = pca.explained_variance_

    kaiser_count = np.sum(eigenvalues > 1)
    print(f"  Kaiser count (λ > 1): {kaiser_count}")

    os.makedirs(output_dir, exist_ok=True)

    # 4-Panel Eigenvalue Analysis
    explained_ratio = plot_eigenvalue_analysis(eigenvalues, folder_name, output_dir, n_top)

    # Top-n energy & frequency plot (original Graph 2)
    n_top_actual = min(n_top, X_pca.shape[1])
    top_scores = X_pca[:, :n_top_actual]
    comp_energies = top_scores**2

    freq_spectra = []
    for i in range(n_top_actual):
        sig = top_scores[:, i] - np.mean(top_scores[:, i])
        psd = np.abs(np.fft.fft(sig))**2
        freq_spectra.append(psd[:len(psd)//2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.bar(range(1, n_top_actual + 1), np.mean(comp_energies, axis=0), color='salmon', alpha=0.7)
    ax1.set_title(f'Mean Energy of Top {n_top_actual} PCA Components - {folder_name}')
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Mean Energy (Score²)')
    ax1.grid(True, axis='y', alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, n_top_actual))
    for i in range(n_top_actual):
        freq_axis = np.linspace(0, 0.5, len(freq_spectra[i]))
        ax2.plot(freq_axis, 10 * np.log10(freq_spectra[i] + 1e-12),
                 label=f'Comp {i+1}', color=colors[i], alpha=0.6)
    ax2.set_title(f'Frequency Response (PSD) of Top {n_top_actual} PCA Scores')
    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Power (dB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{folder_name}_top{n_top}_analysis.png'))
    plt.close()

    print(f"  Graphs saved → {output_dir}")

    # Return top-k eigenvalues for the summary table
    return eigenvalues[:n_top_actual]


# ─────────────────────────────────────────────────────────────────
# Cross-Version Summary Table
# ─────────────────────────────────────────────────────────────────
def build_eigenvalue_table(all_eigenvalues, folders, n_top, output_dir):
    """Create a V1-V20 × Top-k eigenvalue table as CSV and heatmap."""
    import pandas as pd

    # Build DataFrame: rows = versions, cols = PC-1 … PC-k
    col_names = [f'PC-{i+1}' for i in range(n_top)]
    data = {}
    for folder, evals in zip(folders, all_eigenvalues):
        if evals is not None:
            row = list(evals[:n_top])
            # Pad if fewer components returned
            row += [np.nan] * (n_top - len(row))
            data[folder] = row
        else:
            data[folder] = [np.nan] * n_top

    df = pd.DataFrame(data, index=col_names).T   # Versions as rows
    df.index.name = 'Version'

    # Save CSV
    csv_path = os.path.join(output_dir, 'eigenvalue_summary_table.csv')
    df.to_csv(csv_path)
    print(f"\nEigenvalue table saved → {csv_path}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, n_top * 1.2), max(6, len(folders) * 0.5)))
    import matplotlib.colors as mcolors
    # Use log scale for coloring since magnitudes span many orders of magnitude
    log_df = np.log10(df.replace(0, np.nan))
    im = ax.imshow(log_df.values, aspect='auto', cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Eigenvalue)')

    ax.set_xticks(range(n_top))
    ax.set_xticklabels(col_names, rotation=45, ha='right')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_title(f'Top-{n_top} Eigenvalues across Versions (log₁₀ colormap)')

    # Annotate cells
    for r in range(len(df)):
        for c in range(n_top):
            val = df.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f'{val:.2e}', ha='center', va='center',
                        fontsize=6.5, color='white' if np.log10(val+1e-30) < log_df.values.mean() else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvalue_table_heatmap.png'), dpi=150)
    plt.close()
    print(f"Heatmap saved → {output_dir}/eigenvalue_table_heatmap.png")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_TOP = 6    # Number of top components to track

    base_path    = "/media/corsair/New Volume/wirelessdataset/Datasets/"
    results_base = "/media/corsair/New Volume/wirelessdataset/Analysis_Results"
    table_dir    = "/media/corsair/New Volume/wirelessdataset/Analysis_Results/Summary"

    folders = [f for f in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, f)) and f.startswith('V')]
    folders.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    all_eigenvalues = []
    for folder in folders:
        evals = analyze_folder(os.path.join(base_path, folder),
                               os.path.join(results_base, folder),
                               n_top=N_TOP)
        all_eigenvalues.append(evals)

    # Build cross-version summary table
    os.makedirs(table_dir, exist_ok=True)
    build_eigenvalue_table(all_eigenvalues, folders, N_TOP, table_dir)

    print("\nAll Done!")
