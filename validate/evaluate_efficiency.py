"""
Inference utility for the trained efficiency NN.
Loads model and scaler, provides convenience functions for evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

    def efficiency(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def load_efficiency_model(model_path="../train/eff_nn_best.pt",
                          scaler_path="../train/eff_nn_scaler.pkl"):
    """
    Load trained NN model and scaler.
    
    Parameters
    ----------
    model_path : str
        Path to model weights (eff_nn_best.pt)
    scaler_path : str
        Path to StandardScaler pickle (eff_nn_scaler.pkl)
        
    Returns
    -------
    model : EfficiencyNet
        Loaded model on appropriate device
    scaler : StandardScaler
        Fitted feature scaler
    """
    # Load model
    model = EfficiencyNet().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.eval()
    
    # Load scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    return model, scaler


def get_trigger_efficiency(pt, eta, dxy_sig, model, scaler):
    """
    Predict trigger efficiency at arbitrary points in parameter space.
    
    Parameters
    ----------
    pt : float or array-like
        Transverse momentum [GeV]
    eta : float or array-like
        Pseudorapidity (will be converted to absolute value)
    dxy_sig : float or array-like
        Impact parameter significance (will be converted to absolute value)
    model : EfficiencyNet
        Loaded model
    scaler : StandardScaler
        Fitted scaler
        
    Returns
    -------
    eff : float or np.ndarray
        Efficiency in [0, 1]
    """
    pts  = np.atleast_1d(np.asarray(pt, dtype=np.float32))
    etas = np.atleast_1d(np.abs(np.asarray(eta, dtype=np.float32)))
    dxys = np.atleast_1d(np.abs(np.asarray(dxy_sig, dtype=np.float32)))

    # Broadcast to same length
    n = max(len(pts), len(etas), len(dxys))
    pts  = np.broadcast_to(pts,  (n,)).copy()
    etas = np.broadcast_to(etas, (n,)).copy()
    dxys = np.broadcast_to(dxys, (n,)).copy()

    # Stack and scale
    raw = np.column_stack([pts, etas, dxys])
    raw_s = scaler.transform(raw).astype(np.float32)
    
    # Predict
    with torch.no_grad():
        logits = model(to_tensor(raw_s))
        eff = torch.sigmoid(logits).cpu().numpy()
    
    return eff.squeeze() if np.isscalar(pt) and np.isscalar(eta) and np.isscalar(dxy_sig) else eff


if __name__ == "__main__":
    # Example usage
    print(f"Device: {DEVICE}")
    
    model, scaler = load_efficiency_model()
    print("✓ Model loaded successfully")
    
    # Single event
    eff = get_trigger_efficiency(pt=10.5, eta=0.3, dxy_sig=-4.2, model=model, scaler=scaler)
    print(f"\nSingle event efficiency at (pT=10.5, η=0.3, dxy/σ=4.2): {eff:.4f}")
    
    # Batch
    pts = np.array([8.0, 10.0, 15.0])
    etas = np.array([0.1, 0.5, 1.2])
    dxys = np.array([2.5, 3.1, 5.0])
    effs = get_trigger_efficiency(pts, etas, dxys, model, scaler)
    print(f"\nBatch efficiencies: {effs}")