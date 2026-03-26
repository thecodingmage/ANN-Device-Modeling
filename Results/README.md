# ANN-Device-Modeling: TCAD Variability Surrogate Model

This repository implements a multi-branch Artificial Neural Network (ANN) designed to predict the electrical characteristics ($V_{th}$, $I_{off}$, $I_{on}$) of semiconductor devices while accounting for random grain variability. The model serves as a high-speed surrogate for heavy TCAD simulations, specifically capturing subthreshold leakage behavior across multiple decades.

## 1. Project Overview
Semiconductor device performance is significantly impacted by grain configurations (represented as "Seeds") and device geometry (effective width, $W_{eff}$). This project utilizes a **Physics-Aware ANN** that treats "Seeds" as categorical signatures to learn high-dimensional physical configurations rather than treating them as random noise.

## 2. Experimental Results & Observations

During the development process, we trained three different variations of the model to find the optimal architecture.

### Key Finding 1: Architecture Complexity vs. Accuracy
We observed that **increasing model complexity led to a decline in accuracy.** * The **Baseline Model** (simple layers, ReLU activation) achieved the highest precision for $V_{th}$ ($R^2 \approx 0.92$).
* When we introduced a **Refined Model** with higher-dimensional embeddings (20D), **BatchNormalization**, **Dropout**, and **Swish** activation, the $V_{th}$ $R^2$ score dropped to approximately **0.88**.
* **Conclusion:** For this specific dataset size, the simpler architecture generalizes better. Excessive regularization and deeper branches likely introduced noise into the optimization landscape for the sensitive $V_{th}$ parameter.

### Key Finding 2: Sensitivity of $V_{th}$
The $R^2$ score for **Threshold Voltage ($V_{th}$)** showed the most variation across training runs, moving from **0.92** in a lucky single-split run to **0.87** in the final aggregated 5-Fold Cross-Validation. This is expected, as $V_{th}$ is highly dependent on local grain patterns represented by the "Seeds."

### Key Finding 3: Stability of $I_{off}$ and $I_{on}$
Despite the variations in $V_{th}$ and changes in architecture, the predictive accuracy for **$I_{off}$ and $I_{on}$ remained remarkably consistent** across all three models. 
* **$I_{on}$** consistently maintained an $R^2$ score of **~0.999**, showing the model has a perfect grasp of the linear scaling with $W_{eff}$.
* **$I_{off}$** consistently maintained an $R^2$ score of **~0.95 - 0.96**, proving that the log-transformation strategy effectively captures subthreshold leakage regardless of architecture tweaks.

---

## 3. Detailed Performance Summary (Final Validated Model)
The following table represents the aggregated performance across **5-Fold Cross-Validation**, providing the most robust measure of the model's reliability.

| Parameter | RMSE | $R^2$ Score |
| :--- | :--- | :--- |
| **Threshold Voltage ($V_{th}$)** | 0.00488 | **0.87768** |
| **Off-current ($\log I_{off}$)** | 0.07773 | **0.95806** |
| **On-current ($\log I_{on}$)** | 0.00239 | **0.99922** |

---

## 4. Model Architectures Used

### A. Baseline Model (Best $V_{th}$ Result)
```python
# 10D Seed Embedding + ReLU + Dense(64, 32, 16)
emb_seed = layers.Embedding(input_dim=num_seeds, output_dim=10)(input_seed)
x = layers.Dense(64, activation='relu')(merged)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
```

### B. Refined Model (Complexity caused Accuracy Decline)
```python
# 20D Embedding + Swish + BatchNormalization + Dropout
emb_seed = layers.Embedding(input_dim=num_seeds, output_dim=20)(input_seed)
x = layers.Dense(128, activation='swish')(merged)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.1)(x)
```

---

## 5. Key Implementation Details
* **Seed Embedding:** Instead of One-Hot Encoding, we map each unique grain seed to a continuous vector space (Embedding Layer). This allows the model to learn which seeds have similar physical impacts on performance.
* **Logarithmic Scaling:** Both $I_{off}$ and $I_{on}$ are log-transformed ($\log_{10}$) before training to handle the exponential nature of device currents.
* **Validation Strategy:** 5-Fold Cross-Validation was used to ensure that the reported metrics are not dependent on a specific train-test shuffle.

## 6. Repository Structure
* `Notebooks/`: Contains the training scripts and K-fold validation logic.
* `Simulated Dataset/`: The merged TCAD simulation results.
* `Results/`: CSV validation files and scientific correlation plots in $10^x$ notation.
* `ANN Models/`: Saved model weights.

---
*Created by: Nakibul Islam (Executive Member, Machine Learning Club, NIT Silchar)*