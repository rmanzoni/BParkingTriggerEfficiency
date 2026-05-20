"""
3D muon trigger efficiency: f(pT, |eta|, |dxy/sigma_xy|)
Estimated via a neural-network binary classifier trained on tag-and-probe data.

The key idea
────────────
For every probe muon we know whether it fired the HLT (label=1) or not (label=0).
A binary classifier trained on (pT, |eta|, |dxy/sigma_xy|) → {0,1} learns to predict
P(fire | pT, eta, dxy_sig) which IS the trigger efficiency at that point in parameter space.
No binning needed.
"""

import numpy as np
import uproot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. READ DATA
# ══════════════════════════════════════════════════════════════════════════════

# f    = uproot.open("charmonium_partial.root")
f    = uproot.open("aaa.root")
tree = f["tree"]

branches = [
    "mu1_pt", "mu2_pt",
    "mu1_bs_dxy_sig", "mu2_bs_dxy_sig",
    "mu1_eta", "mu2_eta",
    "mass", "vtx_prob", "charge", "dr_12",
    "mu1_id_medium", "mu2_id_medium",
    "mu1_HLT_Mu7p5_Track3p5_Jpsi_probe", "mu2_HLT_Mu7p5_Track3p5_Jpsi_probe",
    "mu1_HLT_Mu7p5_Track3p5_Jpsi_tag",   "mu2_HLT_Mu7p5_Track3p5_Jpsi_tag",
    # prescale OR flags
    "HLT_Mu7_IP4_ps",  "HLT_Mu8_IP3_ps",  "HLT_Mu8_IP5_ps",  "HLT_Mu8_IP6_ps",
    "HLT_Mu8p5_IP3p5_ps", "HLT_Mu9_IP4_ps",  "HLT_Mu9_IP5_ps",  "HLT_Mu9_IP6_ps",
    "HLT_Mu10p5_IP3p5_ps", "HLT_Mu12_IP6_ps",
    # mu1 HLT tag flags
    "mu1_HLT_Mu7_IP4_tag",    "mu1_HLT_Mu8_IP3_tag",    "mu1_HLT_Mu8_IP5_tag",
    "mu1_HLT_Mu8_IP6_tag",    "mu1_HLT_Mu8p5_IP3p5_tag","mu1_HLT_Mu9_IP4_tag",
    "mu1_HLT_Mu9_IP5_tag",    "mu1_HLT_Mu9_IP6_tag",    "mu1_HLT_Mu10p5_IP3p5_tag",
    "mu1_HLT_Mu12_IP6_tag",
    # mu2 HLT tag flags
    "mu2_HLT_Mu7_IP4_tag",    "mu2_HLT_Mu8_IP3_tag",    "mu2_HLT_Mu8_IP5_tag",
    "mu2_HLT_Mu8_IP6_tag",    "mu2_HLT_Mu8p5_IP3p5_tag","mu2_HLT_Mu9_IP4_tag",
    "mu2_HLT_Mu9_IP5_tag",    "mu2_HLT_Mu9_IP6_tag",    "mu2_HLT_Mu10p5_IP3p5_tag",
    "mu2_HLT_Mu12_IP6_tag",
]

arrays = tree.arrays(branches, library="np")
def v(name): return arrays[name]

# ══════════════════════════════════════════════════════════════════════════════
# 2. SELECTIONS  (same as the 2D analysis)
# ══════════════════════════════════════════════════════════════════════════════

ps_or = (
    (v("HLT_Mu7_IP4_ps")      > 0) | (v("HLT_Mu8_IP3_ps")      > 0) |
    (v("HLT_Mu8_IP5_ps")      > 0) | (v("HLT_Mu8_IP6_ps")      > 0) |
    (v("HLT_Mu8p5_IP3p5_ps")  > 0) | (v("HLT_Mu9_IP4_ps")      > 0) |
    (v("HLT_Mu9_IP5_ps")      > 0) | (v("HLT_Mu9_IP6_ps")      > 0) |
    (v("HLT_Mu10p5_IP3p5_ps") > 0) | (v("HLT_Mu12_IP6_ps")     > 0)
)

mu1_tag_hlt = (
    (v("mu1_HLT_Mu7_IP4_tag")       > 0.5) | (v("mu1_HLT_Mu8_IP3_tag")       > 0.5) |
    (v("mu1_HLT_Mu8_IP5_tag")       > 0.5) | (v("mu1_HLT_Mu8_IP6_tag")       > 0.5) |
    (v("mu1_HLT_Mu8p5_IP3p5_tag")   > 0.5) | (v("mu1_HLT_Mu9_IP4_tag")       > 0.5) |
    (v("mu1_HLT_Mu9_IP5_tag")       > 0.5) | (v("mu1_HLT_Mu9_IP6_tag")       > 0.5) |
    (v("mu1_HLT_Mu10p5_IP3p5_tag")  > 0.5) | (v("mu1_HLT_Mu12_IP6_tag")      > 0.5)
)

mu2_tag_hlt = (
    (v("mu2_HLT_Mu7_IP4_tag")       > 0.5) | (v("mu2_HLT_Mu8_IP3_tag")       > 0.5) |
    (v("mu2_HLT_Mu8_IP5_tag")       > 0.5) | (v("mu2_HLT_Mu8_IP6_tag")       > 0.5) |
    (v("mu2_HLT_Mu8p5_IP3p5_tag")   > 0.5) | (v("mu2_HLT_Mu9_IP4_tag")       > 0.5) |
    (v("mu2_HLT_Mu9_IP5_tag")       > 0.5) | (v("mu2_HLT_Mu9_IP6_tag")       > 0.5) |
    (v("mu2_HLT_Mu10p5_IP3p5_tag")  > 0.5) | (v("mu2_HLT_Mu12_IP6_tag")      > 0.5)
)

quality = (
    (np.abs(v("mass") - 3.0969) < 0.1) &
    (v("vtx_prob") > 0.01) &
    (v("charge") == 0) &
    (v("dr_12") > 0.12) &
    ps_or
)

# mu1 as probe
base1 = (
    quality &
    (v("mu2_HLT_Mu7p5_Track3p5_Jpsi_tag")   > 0.5) &
    (v("mu1_HLT_Mu7p5_Track3p5_Jpsi_probe") > 0.5) &
    (np.abs(v("mu1_eta")) < 1.5) &
    v("mu2_id_medium").astype(bool) &
    v("mu1_id_medium").astype(bool)
)

# mu2 as probe
base2 = (
    quality &
    (v("mu1_HLT_Mu7p5_Track3p5_Jpsi_tag")   > 0.5) &
    (v("mu2_HLT_Mu7p5_Track3p5_Jpsi_probe") > 0.5) &
    (np.abs(v("mu2_eta")) < 1.5) &
    v("mu1_id_medium").astype(bool) &
    v("mu2_id_medium").astype(bool)
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD FEATURE MATRIX AND LABELS
# ══════════════════════════════════════════════════════════════════════════════
# Each row = one probe muon candidate
# Features: (pT, |eta|, |dxy_sig|)
# Label:    1 if probe fired the HLT, 0 otherwise

def make_dataset(base_mask, pt_branch, eta_branch, dxy_branch, tag_hlt_mask):
    pt      = v(pt_branch)[base_mask]
    eta_abs = np.abs(v(eta_branch)[base_mask])
    dxy_abs = np.abs(v(dxy_branch)[base_mask])
    label   = tag_hlt_mask[base_mask].astype(np.float32)
    X = np.column_stack([pt, eta_abs, dxy_abs])
    return X, label

X1, y1 = make_dataset(base1, "mu1_pt", "mu1_eta", "mu1_bs_dxy_sig", mu1_tag_hlt)
X2, y2 = make_dataset(base2, "mu2_pt", "mu2_eta", "mu2_bs_dxy_sig", mu2_tag_hlt)

X = np.vstack([X1, X2]).astype(np.float32)
y = np.concatenate([y1, y2]).astype(np.float32)

# Save original (unscaled) features for integrated projections
X_raw = X.copy()

print(f"Total probe candidates : {len(y):,}")
print(f"  of which fired HLT   : {int(y.sum()):,}  ({100*y.mean():.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
val_ds   = TensorDataset(to_tensor(X_val),   to_tensor(y_val))

train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=8192)

# ══════════════════════════════════════════════════════════════════════════════
# 5. NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class EfficiencyNet(nn.Module):
    """
    Small MLP: 3 → [64 → 64 → 32] → 1 (sigmoid).
    The sigmoid output is interpreted directly as efficiency.
    """
    def __init__(self, n_features=3, hidden=(64, 64, 32)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)          # raw logits

    def efficiency(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

model     = EfficiencyNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                        factor=0.5, verbose=True)
criterion = nn.BCEWithLogitsLoss()

# ── Training loop ──────────────────────────────────────────────────────────────
N_EPOCHS    = 100
PATIENCE    = 15   # early stopping

train_losses, val_losses = [], []
best_val_loss  = np.inf
epochs_no_imp  = 0

for epoch in range(1, N_EPOCHS + 1):

    model.train()
    running = 0.0
    for Xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(yb)
    train_loss = running / len(train_ds)

    model.eval()
    running = 0.0
    with torch.no_grad():
        for Xb, yb in val_dl:
            loss = criterion(model(Xb), yb)
            running += loss.item() * len(yb)
    val_loss = running / len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}")

    # early stopping
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        epochs_no_imp = 0
        torch.save(model.state_dict(), "eff_nn_best.pt")
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# Restore best weights
model.load_state_dict(torch.load("eff_nn_best.pt", map_location=DEVICE))
model.eval()
print(f"\nBest validation loss: {best_val_loss:.5f}")

# Save the scaler for later use
with open("eff_nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler to eff_nn_scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONVENIENCE FUNCTION: evaluate efficiency at arbitrary points
# ══════════════════════════════════════════════════════════════════════════════
def predict_efficiency(pt, eta_abs, dxy_abs):
    """
    Predict efficiency at arbitrary points in (pT, |eta|, |dxy_sig|) space.
    
    Parameters
    ----------
    pt, eta_abs, dxy_abs : float or array-like
        Can be scalars or arrays of the same length (or broadcastable).
        
    Returns
    -------
    eff : numpy array of efficiencies in [0, 1]
    """
    pts  = np.atleast_1d(np.asarray(pt,      dtype=np.float32))
    etas = np.atleast_1d(np.asarray(eta_abs, dtype=np.float32))
    dxys = np.atleast_1d(np.asarray(dxy_abs, dtype=np.float32))

    # ── broadcast scalars to match the longest array ──────────────────
    n = max(len(pts), len(etas), len(dxys))
    pts  = np.broadcast_to(pts,  (n,)).copy()
    etas = np.broadcast_to(etas, (n,)).copy()
    dxys = np.broadcast_to(dxys, (n,)).copy()
    # ──────────────────────────────────────────────────────────────────

    raw   = np.column_stack([pts, etas, dxys])
    raw_s = scaler.transform(raw).astype(np.float32)
    
    with torch.no_grad():
        logits = model(to_tensor(raw_s))
        eff = torch.sigmoid(logits).cpu().numpy()
    
    return eff.squeeze() if np.isscalar(pt) and np.isscalar(eta_abs) and np.isscalar(dxy_abs) else eff

# ══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 7a. Training curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses,   label="Validation")
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
ax.set_title("Training history"); ax.legend()
plt.tight_layout()
plt.savefig("eff_nn_training.png", dpi=150)
plt.close()

# ── 7b. 2D slices of the learned efficiency surface ───────────────────────────
# Fix one variable at its median, scan the other two

pt_med  = float(np.median(X[:, 0]))
eta_med = float(np.median(X[:, 1]))
dxy_med = float(np.median(X[:, 2]))

pt_grid  = np.linspace(6,   40,  200)
eta_grid = np.linspace(0,   1.5, 200)
dxy_grid = np.linspace(0,   30,  200)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# pT slice (eta=median, dxy=median)
eff_pt = predict_efficiency(pt_grid, eta_med, dxy_med)
axes[0].plot(pt_grid, eff_pt)
axes[0].set_xlabel(r"$p_T$ [GeV]")
axes[0].set_ylabel("Efficiency")
axes[0].set_title(fr"vs $p_T$ (|$\eta$|={eta_med:.2f}, |dxy/$\sigma$|={dxy_med:.1f})")
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, alpha=0.3)

# eta slice (pt=median, dxy=median)
eff_eta = predict_efficiency(pt_med, eta_grid, dxy_med)
axes[1].plot(eta_grid, eff_eta)
axes[1].set_xlabel(r"|$\eta$|")
axes[1].set_ylabel("Efficiency")
axes[1].set_title(fr"vs |$\eta$| ($p_T$={pt_med:.1f} GeV, |dxy/$\sigma$|={dxy_med:.1f})")
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, alpha=0.3)

# dxy slice (pt=median, eta=median)
eff_dxy = predict_efficiency(pt_med, eta_med, dxy_grid)
axes[2].plot(dxy_grid, eff_dxy)
axes[2].set_xlabel(r"|$d_{xy}/\sigma_{xy}$|")
axes[2].set_ylabel("Efficiency")
axes[2].set_title(fr"vs |dxy/$\sigma$| ($p_T$={pt_med:.1f} GeV, |$\eta$|={eta_med:.2f})")
axes[2].set_ylim(0, 1.05)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eff_nn_slices.png", dpi=150)
plt.close()

# ── 7c. NEW: 1D integrated projections with fine binning ──────────────────────
# pT: 0-30 GeV in 60 bins
# dxy/sigma: 0-15 in 30 bins
# eta: integrated over data distribution at median

print("\nGenerating fine 1D projections...")

pt_grid_fine  = np.linspace(0, 30, 61)  # 60 bins
dxy_grid_fine = np.linspace(0, 15, 31)  # 30 bins
eta_med = float(np.median(X_raw[:, 1]))

# pT projection (integrate over eta and dxy)
eff_pt_1d = predict_efficiency(pt_grid_fine[:-1] + 0.25, eta_med, dxy_med)

# dxy projection (integrate over eta and pt)
eff_dxy_1d = predict_efficiency(pt_med, eta_med, dxy_grid_fine[:-1] + 0.25)

fig1d, axes1d = plt.subplots(1, 2, figsize=(14, 5))
fig1d.suptitle("1D Efficiency projections (slices at median of other dimensions)")

# pT projection
axes1d[0].hist(pt_grid_fine[:-1], bins=pt_grid_fine, weights=eff_pt_1d, 
               edgecolor='black', alpha=0.7, color='steelblue')
axes1d[0].set_xlabel(r"$p_T$ [GeV]")
axes1d[0].set_ylabel("Efficiency")
axes1d[0].set_title(fr"$\varepsilon(p_T)$ at $|\eta|$={eta_med:.2f}, $|d_{{xy}}/\sigma|$={dxy_med:.1f}")
axes1d[0].set_ylim(0, 1.05)
axes1d[0].grid(True, alpha=0.3, axis='y')

# dxy projection
axes1d[1].hist(dxy_grid_fine[:-1], bins=dxy_grid_fine, weights=eff_dxy_1d,
               edgecolor='black', alpha=0.7, color='seagreen')
axes1d[1].set_xlabel(r"$|d_{xy}/\sigma_{xy}|$")
axes1d[1].set_ylabel("Efficiency")
axes1d[1].set_title(fr"$\varepsilon(|d_{{xy}}/\sigma|)$ at $p_T$={pt_med:.1f} GeV, $|\eta|$={eta_med:.2f}")
axes1d[1].set_ylim(0, 1.05)
axes1d[1].grid(True, alpha=0.3, axis='y')

fig1d.tight_layout()
fig1d.savefig("eff_nn_1d_fine_projections.png", dpi=150)
plt.close()

# ── 7d. 2D pT vs dxy map with reference binning ────────────────────────────────
# Use the exact bin edges from your reference histogram
pt_bins  = np.array([6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 10.5, 11.0, 12.0, 20.0, 100.0])
dxy_bins = np.array([0.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 500.0])

n_pt_bins  = len(pt_bins)  - 1
n_dxy_bins = len(dxy_bins) - 1

print(f"Building 2D map with {n_pt_bins} pT bins × {n_dxy_bins} dxy bins...")

# Compute efficiency at bin centers
eff_2d_map = np.zeros((n_dxy_bins, n_pt_bins))

for i_pt, (pt_lo, pt_hi) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
    pt_center = 0.5 * (pt_lo + pt_hi)
    for i_dxy, (dxy_lo, dxy_hi) in enumerate(zip(dxy_bins[:-1], dxy_bins[1:])):
        dxy_center = 0.5 * (dxy_lo + dxy_hi)
        eff_2d_map[i_dxy, i_pt] = predict_efficiency(pt_center, eta_med, dxy_center)

# Create figure matching reference style
fig2d, ax2d = plt.subplots(figsize=(16, 10))

# Create heatmap
im = ax2d.pcolormesh(np.arange(n_pt_bins + 1), np.arange(n_dxy_bins + 1), 
                      eff_2d_map, cmap="viridis", vmin=0, vmax=1, shading="flat")
cbar = fig2d.colorbar(im, ax=ax2d, label="Efficiency")

# Add text annotations in each cell
for i_pt in range(n_pt_bins):
    for i_dxy in range(n_dxy_bins):
        val = eff_2d_map[i_dxy, i_pt]
        ax2d.text(i_pt + 0.5, i_dxy + 0.5, f"{val:.3f}",
                  ha="center", va="center", fontsize=9,
                  color="white" if val < 0.5 else "black",
                  fontweight="bold")

# Set tick labels with bin ranges
pt_labels = [f"[{pt_bins[i]:.4g}, {pt_bins[i+1]:.4g})" for i in range(n_pt_bins)]
dxy_labels = [f"[{dxy_bins[i]:.4g}, {dxy_bins[i+1]:.4g})" for i in range(n_dxy_bins)]

ax2d.set_xticks(np.arange(n_pt_bins) + 0.5)
ax2d.set_yticks(np.arange(n_dxy_bins) + 0.5)
ax2d.set_xticklabels(pt_labels, rotation=45, ha="right", fontsize=10)
ax2d.set_yticklabels(dxy_labels, fontsize=10)

# Grid
ax2d.set_xticks(np.arange(n_pt_bins + 1), minor=True)
ax2d.set_yticks(np.arange(n_dxy_bins + 1), minor=True)
ax2d.grid(which="minor", color="white", linewidth=1.0)
ax2d.tick_params(which="minor", length=0)

ax2d.set_xlabel(r"probe $p_T$ [GeV]", fontsize=12, fontweight="bold")
ax2d.set_ylabel(r"probe $|d_{xy}/\sigma_{xy}|$", fontsize=12, fontweight="bold")
ax2d.set_title(f"NN Trigger Efficiency (tag & probe) - |η|={eta_med:.2f}", 
               fontsize=14, fontweight="bold")
ax2d.set_xlim(0, n_pt_bins)
ax2d.set_ylim(0, n_dxy_bins)

fig2d.tight_layout()
fig2d.savefig("eff_nn_2d_reference_binning.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved validation plots:")
print("  eff_nn_training.png              — loss curves")
print("  eff_nn_slices.png                — 1D slices (fixed medians)")
print("  eff_nn_1d_fine_projections.png   — 1D fine binned projections")
print("  eff_nn_2d_reference_binning.png  — 2D pT×dxy map (comparable to reference)")
print("\nUse predict_efficiency(pt, eta_abs, dxy_abs) to evaluate at any point.")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers for integrated (marginalised) projections
# ─────────────────────────────────────────────────────────────────────────────

def predict_1d_integrated(var_idx: int,
                          var_grid: np.ndarray,
                          X_data: np.ndarray) -> np.ndarray:
    """
    1-D marginalised efficiency.
    Scans column `var_idx` over `var_grid`; integrates the other two
    columns by averaging over all rows in X_data.
    
    Parameters
    ----------
    var_idx : int
        Column index to scan (0=pT, 1=|eta|, 2=|dxy_sig|)
    var_grid : np.ndarray
        1D array of values to scan over
    X_data : np.ndarray
        Original unscaled feature matrix, shape (N, 3)
        
    Returns
    -------
    np.ndarray
        Efficiency values, shape (len(var_grid),)
    """
    n_grid = len(var_grid)
    n_data = len(X_data)

    # Tile: for every grid value repeat all data rows  →  (n_grid*n_data, 3)
    X_eval = np.tile(X_data, (n_grid, 1)).astype(np.float32)
    X_eval[:, var_idx] = np.repeat(var_grid.astype(np.float32), n_data)

    X_s = scaler.transform(X_eval).astype(np.float32)
    with torch.no_grad():
        logits = model(to_tensor(X_s)).squeeze()
        out = torch.sigmoid(logits).cpu().numpy()

    return out.reshape(n_grid, n_data).mean(axis=1)


def predict_2d_integrated(var_idx0: int,
                          var_idx1: int,
                          grid0: np.ndarray,
                          grid1: np.ndarray,
                          X_data: np.ndarray,
                          n_samples: int = 2000) -> np.ndarray:
    """
    2-D marginalised efficiency.
    Scans (var_idx0, var_idx1) over grid0×grid1; integrates the remaining
    column by averaging over `n_samples` rows drawn from X_data.
    
    Parameters
    ----------
    var_idx0, var_idx1 : int
        Column indices to scan (0=pT, 1=|eta|, 2=|dxy_sig|)
    grid0, grid1 : np.ndarray
        1D arrays of values to scan over
    X_data : np.ndarray
        Original unscaled feature matrix, shape (N, 3)
    n_samples : int, optional
        Number of samples to use for integration axis (default 2000)
        
    Returns
    -------
    np.ndarray
        2D efficiency grid, shape (len(grid0), len(grid1))
    """
    var_idx2 = list({0, 1, 2} - {var_idx0, var_idx1})[0]

    # Sub-sample the integration axis for speed / memory
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X_data), size=min(n_samples, len(X_data)), replace=False)
    X_sub = X_data[idx].astype(np.float32)          # (ns, 3)

    n0, n1, ns = len(grid0), len(grid1), len(X_sub)

    # Build (n0*n1*ns, 3) evaluation matrix
    X_eval = np.tile(X_sub, (n0 * n1, 1))           # repeat sub-sample for each grid cell
    X_eval[:, var_idx0] = np.repeat(grid0.astype(np.float32), n1 * ns)
    X_eval[:, var_idx1] = np.tile(
        np.repeat(grid1.astype(np.float32), ns), n0
    )

    X_s = scaler.transform(X_eval).astype(np.float32)
    with torch.no_grad():
        logits = model(to_tensor(X_s)).squeeze()
        out = torch.sigmoid(logits).cpu().numpy()

    return out.reshape(n0, n1, ns).mean(axis=2)   # marginalise over integration axis


# ─────────────────────────────────────────────────────────────────────────────
# Grids for fully integrated projections
# ─────────────────────────────────────────────────────────────────────────────
N1D = 200
N2D = 60    # per axis for 2-D maps

pt_grid_full  = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), N1D)
eta_grid_full = np.linspace(X_raw[:, 1].min(), X_raw[:, 1].max(), N1D)
dxy_grid_full = np.linspace(X_raw[:, 2].min(), X_raw[:, 2].max(), N1D)

pt_grid2  = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), N2D)
eta_grid2 = np.linspace(X_raw[:, 1].min(), X_raw[:, 1].max(), N2D)
dxy_grid2 = np.linspace(X_raw[:, 2].min(), X_raw[:, 2].max(), N2D)

# ─────────────────────────────────────────────────────────────────────────────
# 1-D fully integrated projections
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing 1-D fully integrated projections …")
eff_pt_int  = predict_1d_integrated(0, pt_grid_full,  X_raw)
eff_eta_int = predict_1d_integrated(1, eta_grid_full, X_raw)
eff_dxy_int = predict_1d_integrated(2, dxy_grid_full, X_raw)

fig1d_full, axes1d_full = plt.subplots(1, 3, figsize=(15, 4))
fig1d_full.suptitle("1-D fully integrated efficiency (other dims marginalised over data)")

axes1d_full[0].plot(pt_grid_full,  eff_pt_int,  color="steelblue", linewidth=2)
axes1d_full[0].set_xlabel(r"$p_T$ [GeV]");  axes1d_full[0].set_ylabel("Efficiency")
axes1d_full[0].set_title(r"$\varepsilon(p_T)$, integrated over $|\eta|$, $|d_{xy}/\sigma|$")
axes1d_full[0].set_ylim(0, 1.05)
axes1d_full[0].grid(True, alpha=0.3)

axes1d_full[1].plot(eta_grid_full, eff_eta_int, color="darkorange", linewidth=2)
axes1d_full[1].set_xlabel(r"$|\eta|$");     axes1d_full[1].set_ylabel("Efficiency")
axes1d_full[1].set_title(r"$\varepsilon(|\eta|)$, integrated over $p_T$, $|d_{xy}/\sigma|$")
axes1d_full[1].set_ylim(0, 1.05)
axes1d_full[1].grid(True, alpha=0.3)

axes1d_full[2].plot(dxy_grid_full, eff_dxy_int, color="seagreen", linewidth=2)
axes1d_full[2].set_xlabel(r"$|d_{xy}/\sigma|$"); axes1d_full[2].set_ylabel("Efficiency")
axes1d_full[2].set_title(r"$\varepsilon(|d_{xy}/\sigma|)$, integrated over $p_T$, $|\eta|$")
axes1d_full[2].set_ylim(0, 1.05)
axes1d_full[2].grid(True, alpha=0.3)

fig1d_full.tight_layout()
fig1d_full.savefig("efficiency_1d_integrated.pdf", dpi=150)
plt.close()



print("\nComputing 1-D fully integrated projections …")

pt_grid_alt  = np.linspace(0, 30, 61)
dxy_grid_alt = np.linspace(0, 15, 31)

eff_pt_int_alt  = predict_1d_integrated(0, pt_grid_alt,  X_raw)
eff_dxy_int_alt = predict_1d_integrated(2, dxy_grid_alt, X_raw)

fig1d_full, axes1d_full = plt.subplots(1, 3, figsize=(15, 4))
fig1d_full.suptitle("1-D fully integrated efficiency (other dims marginalised over data)")

axes1d_full[0].plot(pt_grid_alt,  eff_pt_int_alt,  color="steelblue", linewidth=2)
axes1d_full[0].set_xlabel(r"$p_T$ [GeV]");  axes1d_full[0].set_ylabel("Efficiency")
axes1d_full[0].set_title(r"$\varepsilon(p_T)$, integrated over $|\eta|$, $|d_{xy}/\sigma|$")
axes1d_full[0].set_ylim(0, .45)
axes1d_full[0].grid(True, alpha=0.3)

axes1d_full[1].plot(eta_grid_full, eff_eta_int, color="darkorange", linewidth=2)
axes1d_full[1].set_xlabel(r"$|\eta|$");     axes1d_full[1].set_ylabel("Efficiency")
axes1d_full[1].set_title(r"$\varepsilon(|\eta|)$, integrated over $p_T$, $|d_{xy}/\sigma|$")
axes1d_full[1].set_ylim(0, .15)
axes1d_full[1].grid(True, alpha=0.3)

axes1d_full[2].plot(dxy_grid_alt, eff_dxy_int_alt, color="seagreen", linewidth=2)
axes1d_full[2].set_xlabel(r"$|d_{xy}/\sigma|$"); axes1d_full[2].set_ylabel("Efficiency")
axes1d_full[2].set_title(r"$\varepsilon(|d_{xy}/\sigma|)$, integrated over $p_T$, $|\eta|$")
axes1d_full[2].set_ylim(0, .45)
axes1d_full[2].grid(True, alpha=0.3)

fig1d_full.tight_layout()
fig1d_full.savefig("efficiency_1d_integrated_alt_binning.pdf", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2-D integrated projections
# ─────────────────────────────────────────────────────────────────────────────
print("Computing 2-D integrated projections …")


eff_pt_eta  = predict_2d_integrated(0, 1, pt_grid_alt  , eta_grid_full, X_raw)   # integrate dxy
eff_pt_dxy  = predict_2d_integrated(0, 2, pt_grid_alt  , dxy_grid_alt , X_raw)   # integrate eta
eff_eta_dxy = predict_2d_integrated(1, 2, eta_grid_full, dxy_grid_alt , X_raw) # integrate pt

fig2d_full, axes2d_full = plt.subplots(1, 3, figsize=(18, 5))
fig2d_full.suptitle("2-D integrated efficiency (remaining dim marginalised over data)")

cmap = "viridis"
common_kw = dict(origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)

def _extent(xg, yg):
    return [xg[0], xg[-1], yg[0], yg[-1]]

im0 = axes2d_full[0].imshow(eff_pt_eta.T,
                        extent=_extent(pt_grid_alt, eta_grid_full), **common_kw)
axes2d_full[0].set_xlabel(r"$p_T$ [GeV]")
axes2d_full[0].set_ylabel(r"$|\eta|$")
axes2d_full[0].set_title(r"$\varepsilon(p_T,\,|\eta|)$, integrated over $|d_{xy}/\sigma|$")
fig2d_full.colorbar(im0, ax=axes2d_full[0], label="Efficiency")

im1 = axes2d_full[1].imshow(eff_pt_dxy.T,
                        extent=_extent(pt_grid_alt, dxy_grid_alt), **common_kw)
axes2d_full[1].set_xlabel(r"$p_T$ [GeV]")
axes2d_full[1].set_ylabel(r"$|d_{xy}/\sigma|$")
axes2d_full[1].set_title(r"$\varepsilon(p_T,\,|d_{xy}/\sigma|)$, integrated over $|\eta|$")
fig2d_full.colorbar(im1, ax=axes2d_full[1], label="Efficiency")

im2 = axes2d_full[2].imshow(eff_eta_dxy.T,
                        extent=_extent(eta_grid_full, dxy_grid_alt), **common_kw)
axes2d_full[2].set_xlabel(r"$|\eta|$")
axes2d_full[2].set_ylabel(r"$|d_{xy}/\sigma|$")
axes2d_full[2].set_title(r"$\varepsilon(|\eta|,\,|d_{xy}/\sigma|)$, integrated over $p_T$")
fig2d_full.colorbar(im2, ax=axes2d_full[2], label="Efficiency")

for ax in axes2d_full:
    ax.grid(False)

fig2d_full.tight_layout()
fig2d_full.savefig("efficiency_2d_integrated_alt.pdf", dpi=150)
plt.close()

##########################################################################################

print("Computing 2-D integrated projections …")
eff_pt_eta  = predict_2d_integrated(0, 1, pt_grid2, eta_grid2, X_raw)   # integrate dxy
eff_pt_dxy  = predict_2d_integrated(0, 2, pt_grid2, dxy_grid2, X_raw)   # integrate eta
eff_eta_dxy = predict_2d_integrated(1, 2, eta_grid2, dxy_grid2, X_raw) # integrate pt

fig2d_full, axes2d_full = plt.subplots(1, 3, figsize=(18, 5))
fig2d_full.suptitle("2-D integrated efficiency (remaining dim marginalised over data)")

cmap = "viridis"
common_kw = dict(origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)

def _extent(xg, yg):
    return [xg[0], xg[-1], yg[0], yg[-1]]

im0 = axes2d_full[0].imshow(eff_pt_eta.T,
                        extent=_extent(pt_grid2, eta_grid2), **common_kw)
axes2d_full[0].set_xlabel(r"$p_T$ [GeV]")
axes2d_full[0].set_ylabel(r"$|\eta|$")
axes2d_full[0].set_title(r"$\varepsilon(p_T,\,|\eta|)$, integrated over $|d_{xy}/\sigma|$")
fig2d_full.colorbar(im0, ax=axes2d_full[0], label="Efficiency")

im1 = axes2d_full[1].imshow(eff_pt_dxy.T,
                        extent=_extent(pt_grid2, dxy_grid2), **common_kw)
axes2d_full[1].set_xlabel(r"$p_T$ [GeV]")
axes2d_full[1].set_ylabel(r"$|d_{xy}/\sigma|$")
axes2d_full[1].set_title(r"$\varepsilon(p_T,\,|d_{xy}/\sigma|)$, integrated over $|\eta|$")
fig2d_full.colorbar(im1, ax=axes2d_full[1], label="Efficiency")

im2 = axes2d_full[2].imshow(eff_eta_dxy.T,
                        extent=_extent(eta_grid2, dxy_grid2), **common_kw)
axes2d_full[2].set_xlabel(r"$|\eta|$")
axes2d_full[2].set_ylabel(r"$|d_{xy}/\sigma|$")
axes2d_full[2].set_title(r"$\varepsilon(|\eta|,\,|d_{xy}/\sigma|)$, integrated over $p_T$")
fig2d_full.colorbar(im2, ax=axes2d_full[2], label="Efficiency")

for ax in axes2d_full:
    ax.grid(False)

fig2d_full.tight_layout()
fig2d_full.savefig("efficiency_2d_integrated.pdf", dpi=150)
plt.close()

print("\nAll plots saved!")
