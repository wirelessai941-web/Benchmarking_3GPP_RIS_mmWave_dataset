import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import scipy.io

def get_mat_data(file_path, sample_limit=500):
    """Load a subset of data from a .mat file."""
    try:
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if key not in ['#refs#', '#subsystem#']:
                    # MATLAB v7.3 stores arrays transposed: (Features, Samples)
                    ds = f[key]
                    n_samples = ds.shape[1]
                    limit = min(sample_limit, n_samples)
                    data = np.array(ds[:, :limit])
                    return data.T # Return as (Samples, Features)
    except Exception:
        try:
            data = scipy.io.loadmat(file_path)
            for key in data.keys():
                if not key.startswith('__'):
                    d = data[key]
                    limit = min(sample_limit, d.shape[0])
                    return d[:limit, :]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return None

def explore_features(base_path, output_dir, samples_per_folder=100):
    os.makedirs(output_dir, exist_ok=True)
    
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.startswith('V')]
    folders.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    
    all_stats = []
    sampled_X = []
    channel_names = ['BS_RIS', 'RIS_UE', 'Direct']
    
    print(f"Sampling {samples_per_folder} samples from {len(folders)} folders...")
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        files = os.listdir(folder_path)
        
        # Identify files
        f_br = [f for f in files if 'BS_RIS_channel' in f]
        f_ru = [f for f in files if 'RIS_UE_channel' in f]
        f_di = [f for f in files if 'direct_channel' in f]
        
        if not (f_br and f_ru and f_di): continue
        
        # Load small sets
        d_br = get_mat_data(os.path.join(folder_path, f_br[0]), samples_per_folder)
        d_ru = get_mat_data(os.path.join(folder_path, f_ru[0]), samples_per_folder)
        d_di = get_mat_data(os.path.join(folder_path, f_di[0]), samples_per_folder)
        
        if d_br is None or d_ru is None or d_di is None: continue
        
        # Basic Stats
        for name, d in zip(channel_names, [d_br, d_ru, d_di]):
            mag = np.abs(d)
            all_stats.append({
                'Folder': folder,
                'Channel': name,
                'Mean': np.mean(mag),
                'Std': np.std(mag),
                'Max': np.max(mag),
                'Energy_Mean': np.mean(mag**2)
            })
            
        # Concatenate for global exploration (flattening)
        flat_br = d_br.reshape(d_br.shape[0], -1)
        flat_ru = d_ru.reshape(d_ru.shape[0], -1)
        flat_di = d_di.reshape(d_di.shape[0], -1)
        
        # Combined feature vector for these samples
        X_folder = np.hstack([np.abs(flat_br), np.abs(flat_ru), np.abs(flat_di)])
        sampled_X.append(X_folder)

    # Convert stats to readable format
    import pandas as pd
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv(os.path.join(output_dir, 'feature_stats.csv'), index=False)
    print(f"Saved stats to {output_dir}/feature_stats.csv")

    # Combined Analysis
    X_all = np.vstack(sampled_X)
    print(f"Combined exploratory set shape: {X_all.shape}")

    # 1. Channel Scale Comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Channel', y='Mean', data=df_stats)
    plt.title('Distribution of Mean Magnitudes Across Versions')
    plt.savefig(os.path.join(output_dir, 'channel_scales.png'))
    plt.close()

    # 2. Correlation between Channel Energies
    # We take the mean energy of each channel per folder
    pivot_energy = df_stats.pivot(index='Folder', columns='Channel', values='Energy_Mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_energy.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation between Channel Energies')
    plt.savefig(os.path.join(output_dir, 'channel_correlation.png'))
    plt.close()

    # 3. High-level Distribution of all sampled features
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(channel_names):
        # Sample some features from X_all to show distribution
        # (Using a very small slice for speed)
        samples = X_all[:, i*1000 : (i+1)*1000].flatten() if X_all.shape[1] > (i+1)*1000 else X_all.flatten()
        sns.kdeplot(samples, label=name, shade=True)
    plt.title('Distribution of Feature Magnitudes (Sampled)')
    plt.xlabel('Magnitude')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()

    print(f"Exploration Complete! Results saved to {output_dir}")

if __name__ == "__main__":
    base_path = "/media/corsair/New Volume/wirelessdataset/Datasets/"
    output_dir = "/media/corsair/New Volume/wirelessdataset/Exploration_Results"
    explore_features(base_path, output_dir, samples_per_folder=100)
