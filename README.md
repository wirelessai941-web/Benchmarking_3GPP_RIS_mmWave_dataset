# Benchmarking_3GPP_RIS_mmWave_dataset

## 📌 Overview

A large-scale GPP TR 38.901-compliant dataset designed for research in **Reconfigurable Intelligent Surface (RIS)**-assisted wireless communication systems. It follows standardized channel modeling principles inspired by 3GPP TR 38.901 and supports machine learning tasks such as:

* Channel Quality Indicator (CQI) prediction
* Beamforming optimization
* RIS configuration learning
* Domain generalization under distribution shift

The dataset contains **200,000 labeled channel realizations** across **20 deployment variants (V1–V20)** at 28 GHz.

---

## 📂 Dataset Structure

USe Dataset_generation/RIS_mmWave_MultiDataset_Pipeline_V3_NoiseVariants.ipynb to generate data samples

```
dataset/
│── data/
│    ├── V1/
│    ├── V2/
│    └── V20/
|_____metadata.csv
│── README.md
```

---

## 📊 Data Description

Each sample corresponds to a **RIS-aided wireless channel realization**.

### Features

* 36-dimensional statistical feature vector derived from:

  * BS–RIS channel
  * RIS–UE channel
  * Direct BS–UE channel

Extracted statistics include:

* Mean, standard deviation
* Max, min, median
* Energy
* Skewness, kurtosis
* Percentiles (25th, 75th)
* Range, log-energy

### Labels

* CQI index / Beam index (depending on task setup)

---

## 🔀 Dataset Splits

### 1. IID Split

* Train: 80
* Validation: 20%

File: `splits/iid_split.csv`


---

## 🧪 Use Cases

The dataset is intended for:

* RIS-aided beamforming
* Wireless channel prediction
* Domain adaptation and generalization
* Reinforcement learning for RIS control
* Robust learning under distribution shift

---

## ⚠️ Data Limitations

* Simulation-based dataset (no real hardware measurements)
* No hardware impairments:

  * Phase noise
  * Quantization errors
  * RF non-linearities
* Static user positions (no mobility modeling)
* Fixed blockage conditions

Future versions will include:

* Hardware-in-the-loop measurements
* Mobility-aware channel evolution
* Dynamic blockage modeling

---

## ⚖️ Data Biases

* Scenario imbalance across variants (e.g., UMa vs Indoor)
* Limited representation of extreme propagation conditions
* Synthetic label generation may introduce modeling bias

These factors may affect generalization to real-world deployments.

---

## 🔐 Personal & Sensitive Information

This dataset contains:

* ❌ No personal data
* ❌ No demographic information
* ❌ No location traces

It consists entirely of **synthetic wireless channel data**.

---

## 🌍 Social Impact

### Positive

* Enables efficient wireless system design
* Supports research in 6G technologies
* Improves spectral efficiency and connectivity

### Risks

* Direct deployment without real-world validation may degrade performance
* Potential fairness issues if models generalize poorly across environments

---

## 🔬 Data Generation & Provenance

### Source Models

* 3GPP TR 38.901 channel models
* DeepMIMO simulation framework

### Pipeline

1. Scenario configuration (UMa, UMi, Indoor)
2. Channel simulation
3. RIS interaction modeling
4. Feature extraction (36-dim statistics)
5. Label generation (CQI / beam index)

---

## 📦 Croissant Metadata

This dataset includes a **MLCommons Croissant file**:

```
dataset.croissant.json
```

It provides:

* Machine-readable schema
* File mappings
* Metadata for reproducibility

---

## 🚀 Getting Started

### Load metadata

```python
import pandas as pd

df = pd.read_csv("metadata.csv")
print(df.head())
```

---

### Load a sample

```python

def load_data_folder(folder):
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

if __name__ == "__main__":

  for folder in Data_FOLDERS:
          X, y = load_data_folder(folder)
          X_list.append(X)
          y_list.append(y)

---

## 📜 License

This dataset is released under:

**Creative Commons Attribution 4.0 (CC BY 4.0)**

---

## 📖 Citation

```
@dataset{miran_2026,
  author = {First, second and thiird},
  title = {MiRaN: RIS-Aided Wireless Dataset for B5G/6G},
  year = {2026}
}
```

