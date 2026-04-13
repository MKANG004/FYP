#!/usr/bin/env python3
"""
Pure Cartesian Model - Fast Version
Quick training script optimized for large datasets.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import random, os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CartesianSample:
    def __init__(self, z, pos, y):
        self.z = z
        self.pos = pos
        self.y = y


def fractional_to_cartesian(frac_coords, a, b, c, alpha, beta, gamma):
    """Convert fractional to Cartesian coordinates."""
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)
    
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    
    v1 = np.array([a, 0, 0])
    v2 = np.array([b * cos_gamma, b * sin_gamma, 0])
    
    v3_x = c * cos_beta
    v3_y = c * (cos_alpha - cos_beta * cos_gamma) / (sin_gamma + 1e-10)
    v3_z_sq = c**2 - v3_x**2 - v3_y**2
    v3_z = np.sqrt(max(0, v3_z_sq))
    v3 = np.array([v3_x, v3_y, v3_z])
    
    lattice_matrix = np.array([v1, v2, v3]).T
    frac_coords = np.array(frac_coords)
    cart_coords = frac_coords @ lattice_matrix.T
    return cart_coords


def parse_sites_json(sites_json, a, b, c, alpha, beta, gamma):
    """Parse JSON sites and convert to Cartesian."""
    ELEMENT_MAP = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
        'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    }
    
    try:
        sites = json.loads(sites_json)
    except:
        return None, None
    
    elements, frac_coords = [], []
    for site in sites:
        elem_str = site.get("element", "")
        if not elem_str:
            continue
        z_num = ELEMENT_MAP.get(elem_str, None)
        if z_num is None:
            continue
        elements.append(z_num)
        frac_coords.append([
            float(site.get("f_x", 0)),
            float(site.get("f_y", 0)),
            float(site.get("f_z", 0))
        ])
    
    if not elements:
        return None, None
    
    try:
        cart_coords = fractional_to_cartesian(frac_coords, a, b, c, alpha, beta, gamma)
        z = torch.tensor(elements, dtype=torch.long)
        pos = torch.tensor(cart_coords, dtype=torch.float32)
        return z, pos
    except:
        return None, None


def load_cartesian_dataset_from_csv(csv_path: Path, target_col: str = "band_gap", max_samples: int = None):
    """Load Cartesian dataset from CSV."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter valid rows
    df = df.dropna(subset=[target_col, "sites", "a", "b", "c"]).reset_index(drop=True)
    
    if max_samples:
        df = df.head(max_samples)
    
    print(f"Found {len(df)} potential samples")
    
    samples = []
    skipped = 0
    
    for idx, row in df.iterrows():
        if (idx + 1) % 5000 == 0:
            print(f"Processing {idx + 1}/{len(df)}... ({len(samples)} loaded so far)")
        
        try:
            a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
            alpha = float(row.get("alpha", 90.0))
            beta = float(row.get("beta", 90.0))
            gamma = float(row.get("gamma", 90.0))
            
            if a <= 0 or b <= 0 or c <= 0:
                skipped += 1
                continue
            
            sites_json = str(row["sites"])
            z, pos = parse_sites_json(sites_json, a, b, c, alpha, beta, gamma)
            
            if z is None or pos is None:
                skipped += 1
                continue
            
            y = torch.tensor([float(row[target_col])], dtype=torch.float32)
            samples.append(CartesianSample(z, pos, y))
            
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"Warning: Skipped row {idx}: {str(e)[:60]}")
    
    print(f"Successfully loaded {len(samples)} samples (skipped {skipped})")
    return samples


class PureCartesianModel(nn.Module):
    """Simple Cartesian model."""
    def __init__(self, hidden_dim=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(120, hidden_dim)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.atom_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for i in range(num_layers)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, z, pos):
        h_elem = self.embed(z)
        center = pos.mean(dim=0, keepdim=True)
        pos_centered = pos - center
        h_pos = self.pos_encoder(pos_centered)
        h = torch.cat([h_elem, h_pos], dim=-1)
        
        h = self.atom_layers[0](h)
        for layer in self.atom_layers[1:]:
            h = h + layer(h)
        
        h_global = h.mean(dim=0, keepdim=True)
        out = self.output_head(h_global)
        return out


def split_dataset(samples, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    val_size = max(1, int(len(samples) * val_frac))
    train = [samples[i] for i in idxs[val_size:]]
    val = [samples[i] for i in idxs[:val_size]]
    return train, val


def mae(y_true, y_pred):
    return float(torch.mean(torch.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser(description="Pure Cartesian model (fast version)")
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--target", type=str, default="band_gap")
    ap.add_argument("--max-samples", type=int, default=None, help="Max samples to load (None = all data)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    print(f"Using device: {device}\n")

    out_dir = Path(args.output or (Path("runs") / f"cartesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    samples = load_cartesian_dataset_from_csv(csv_path, args.target, max_samples=args.max_samples)
    print()
    
    train, val = split_dataset(samples, val_frac=0.2, seed=args.seed)
    print(f"Train: {len(train)}, Val: {len(val)}\n")

    model = PureCartesianModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    history = {"epoch": [], "val_mae": [], "val_rmse": [], "val_r2": []}
    best = float("inf")

    print("Starting training...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_start in range(0, len(train), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(train))
            batch = train[batch_start:batch_end]
            
            for s in batch:
                z, pos, y = s.z.to(device), s.pos.to(device), s.y.to(device)
                pred = model(z, pos)
                loss = criterion(pred.view_as(y), y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for s in val:
                z, pos, y = s.z.to(device), s.pos.to(device), s.y.to(device)
                pred = model(z, pos)
                ys.append(y.cpu())
                ps.append(pred.cpu())

        y_true = torch.cat(ys)
        y_pred = torch.cat(ps)
        vmae = mae(y_true, y_pred)
        vrmse = rmse(y_true, y_pred)
        vr2 = r2_score(y_true, y_pred)
        
        history["epoch"].append(epoch)
        history["val_mae"].append(vmae)
        history["val_rmse"].append(vrmse)
        history["val_r2"].append(vr2)
        
        train_loss_avg = train_loss / len(train)
        print(f"Epoch {epoch:3d} | loss: {train_loss_avg:.4f} | MAE: {vmae:.4f} | RMSE: {vrmse:.4f} | R²: {vr2:.3f}")

        if vmae < best:
            best = vmae
            torch.save({"model_state": model.state_dict()}, out_dir / "model.pt")

    # Save results
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(history, f, indent=2)
    
    cfg = {
        "csv": str(csv_path),
        "target": args.target,
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with (out_dir / "config_resolved.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[SUCCESS] Training complete!")
    print(f"[SUCCESS] Saved to: {out_dir}")
    print(f"[SUCCESS] Final MAE: {history['val_mae'][-1]:.4f}, RMSE: {history['val_rmse'][-1]:.4f}, R2: {history['val_r2'][-1]:.3f}")


if __name__ == "__main__":
    main()
