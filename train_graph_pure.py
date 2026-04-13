#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Element to atomic number map (no pymatgen needed)
ELEMENT_MAP = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
}


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


@dataclass
class GraphSample:
    x: torch.Tensor        # [N] long (atomic numbers)
    pos: torch.Tensor      # [N,3]
    edge_index: torch.Tensor  # [2,E]
    y: torch.Tensor        # [1]


def structure_to_graph(struct: Structure, cutoff: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.tensor([site.specie.number for site in struct.sites], dtype=torch.long)
    pos = torch.tensor(struct.cart_coords, dtype=torch.float32)
    # naive pairwise radius neighbors (small N typical)
    N = pos.shape[0]
    if N <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return z, pos, edge_index
    diffs = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N,N,3]
    dists = torch.linalg.norm(diffs, dim=-1)     # [N,N]
    mask = (dists > 0) & (dists <= cutoff)
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    return z, pos, edge_index


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


def parse_sites_json(sites_json: str, a: float, b: float, c: float, 
                     alpha: float, beta: float, gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse JSON sites and convert fractional coordinates to Cartesian.
    
    Args:
        sites_json: JSON string with sites data
        a, b, c: Lattice parameters (lengths)
        alpha, beta, gamma: Lattice angles (degrees)
    
    Returns:
        z: [N] atomic numbers
        pos: [N, 3] Cartesian coordinates
    """
    try:
        sites = json.loads(sites_json)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)[:100]}")
    
    elements = []
    frac_coords = []
    
    for site in sites:
        element_str = site.get("element", "")
        if not element_str:
            continue
        
        # Convert element string to atomic number using ELEMENT_MAP
        z_num = ELEMENT_MAP.get(element_str, None)
        if z_num is None:
            continue  # Skip unknown elements
        
        elements.append(z_num)
        # Get fractional coordinates
        f_x = float(site.get("f_x", 0.0))
        f_y = float(site.get("f_y", 0.0))
        f_z = float(site.get("f_z", 0.0))
        frac_coords.append([f_x, f_y, f_z])
    
    if len(elements) == 0:
        raise ValueError("No valid sites found")
    
    # Convert fractional to Cartesian coordinates
    cart_coords = fractional_to_cartesian(frac_coords, a, b, c, alpha, beta, gamma)
    
    z = torch.tensor(elements, dtype=torch.long)
    pos = torch.tensor(cart_coords, dtype=torch.float32)
    
    return z, pos


def detect_latest_dataset() -> Tuple[Path, Path]:
    data_root = Path("data").resolve()
    candidates = sorted(data_root.glob("mp_subset_*/materials.parquet"))
    if not candidates:
        raise FileNotFoundError("No Parquet found. Run fetch_mp_bandgaps.py first.")
    pq = candidates[-1]
    cifs_dir = pq.parent / "cifs"
    if not cifs_dir.exists():
        raise FileNotFoundError(f"Missing CIFs directory: {cifs_dir}")
    return pq, cifs_dir


def load_graph_dataset(parquet_path: Path, cifs_dir: Path, cutoff: float) -> List[GraphSample]:
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["band_gap", "material_id"]).reset_index(drop=True)
    samples: List[GraphSample] = []
    for _, row in df.iterrows():
        mpid = str(row["material_id"])
        cif_path = cifs_dir / f"{mpid}.cif"
        if not cif_path.exists():
            continue
        try:
            struct = Structure.from_file(str(cif_path))
            z, pos, edge_index = structure_to_graph(struct, cutoff)
            y = torch.tensor([float(row["band_gap"])], dtype=torch.float32)
            samples.append(GraphSample(x=z, pos=pos, edge_index=edge_index, y=y))
        except Exception:
            continue
    if not samples:
        raise RuntimeError("No graphs created. Check CIFs and parquet alignment.")
    return samples


def load_graph_dataset_from_csv(csv_path: Path, cutoff: float, target_col: str = "band_gap", max_samples: int = None) -> List[GraphSample]:
    """
    Load graph dataset from CSV file with JSON sites column.
    
    Args:
        csv_path: Path to CSV file
        cutoff: Distance cutoff for edges
        target_col: Name of target column (default: "band_gap")
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        List of GraphSample objects
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter rows with valid target and sites
    df = df.dropna(subset=[target_col, "sites"]).reset_index(drop=True)
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
    
    print(f"Found {len(df)} rows with {target_col} and sites data")
    
    samples: List[GraphSample] = []
    skipped = 0
    
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"Processing {idx + 1}/{len(df)}...")
        
        try:
            # Get lattice parameters
            a = float(row.get("a", 0))
            b = float(row.get("b", 0))
            c = float(row.get("c", 0))
            alpha = float(row.get("alpha", 90.0))
            beta = float(row.get("beta", 90.0))
            gamma = float(row.get("gamma", 90.0))
            
            # Skip if lattice parameters are invalid
            if a <= 0 or b <= 0 or c <= 0:
                skipped += 1
                continue
            
            # Parse sites and get atomic numbers + positions
            sites_json = str(row["sites"])
            z, pos = parse_sites_json(sites_json, a, b, c, alpha, beta, gamma)
            
            # Build graph edges
            N = pos.shape[0]
            if N <= 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                diffs = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N,N,3]
                dists = torch.linalg.norm(diffs, dim=-1)     # [N,N]
                mask = (dists > 0) & (dists <= cutoff)
                src, dst = torch.nonzero(mask, as_tuple=True)
                edge_index = torch.stack([src, dst], dim=0)
            
            # Get target value
            y = torch.tensor([float(row[target_col])], dtype=torch.float32)
            
            samples.append(GraphSample(x=z, pos=pos, edge_index=edge_index, y=y))
            
        except Exception as e:
            skipped += 1
            if skipped <= 5:  # Print first few errors
                print(f"Warning: Skipped row {idx}: {e}")
            continue
    
    print(f"Successfully loaded {len(samples)} samples (skipped {skipped})")
    if not samples:
        raise RuntimeError("No graphs created. Check CSV format and data.")
    return samples


class SimpleMPNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, conv_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(120, hidden_dim)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(conv_layers)
        ])
        self.self_lin = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(conv_layers)])
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def message_passing(self, h: torch.Tensor, edge_index: torch.Tensor, layer: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index  # [E], [E]
        src_h = h.index_select(0, src)
        dst_h = h.index_select(0, dst)
        m = self.msg_mlps[layer](torch.cat([src_h, dst_h], dim=-1))  # [E, H]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        h_new = F.relu(self.self_lin[layer](h) + agg)
        return h_new

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.view(-1))
        for l in range(len(self.msg_mlps)):
            h = self.message_passing(h, edge_index, l)
        g = h.mean(dim=0, keepdim=True)  # global mean pool
        return self.out(g)  # [1,1]


def split_dataset(samples: List[GraphSample], val_frac: float, seed: int) -> Tuple[List[GraphSample], List[GraphSample]]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    val_size = max(1, int(len(samples) * val_frac))
    val_idx = idxs[:val_size]
    train_idx = idxs[val_size:]
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
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
    ap = argparse.ArgumentParser(description="Pure-PyTorch graph baseline (no PyG)")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV file with sites data")
    ap.add_argument("--parquet", type=str, default=None)
    ap.add_argument("--cifs", type=str, default=None)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--target", type=str, default="band_gap", help="Target column name")
    ap.add_argument("--max-samples", type=int, default=None, help="Max samples to load (None = all)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    print(f"Using device: {device}")

    out_dir = Path(args.output or (Path("runs") / f"graph_pure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset - prioritize CSV if provided
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        samples = load_graph_dataset_from_csv(csv_path, cutoff=args.cutoff, target_col=args.target, max_samples=args.max_samples)
    elif args.parquet and args.cifs:
        parquet_path = Path(args.parquet)
        cifs_dir = Path(args.cifs)
        samples = load_graph_dataset(parquet_path, cifs_dir, cutoff=args.cutoff)
    else:
        parquet_path, cifs_dir = detect_latest_dataset()
        samples = load_graph_dataset(parquet_path, cifs_dir, cutoff=args.cutoff)
    
    train, val = split_dataset(samples, val_frac=0.2, seed=args.seed)
    print(f"Train: {len(train)}, Val: {len(val)}")

    model = SimpleMPNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    history = {"epoch": [], "val_mae": [], "val_rmse": [], "val_r2": []}
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for s in train:
            x, ei, y = s.x.to(device), s.edge_index.to(device), s.y.to(device)
            pred = model(x, ei)
            loss = criterion(pred.view_as(y), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += loss.item()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for s in val:
                x, ei, y = s.x.to(device), s.edge_index.to(device), s.y.to(device)
                pred = model(x, ei)
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
        print(f"Epoch {epoch:03d} | train_loss: {train_loss_avg:.4f} | val_MAE: {vmae:.4f} | val_RMSE: {vrmse:.4f} | val_R2: {vr2:.3f}")
        
        if vmae < best:
            best = vmae
            torch.save({"model_state": model.state_dict()}, out_dir / "model.pt")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    cfg = {
        "csv": str(args.csv) if args.csv else None,
        "parquet": str(args.parquet) if args.parquet else None,
        "cifs": str(args.cifs) if args.cifs else None,
        "cutoff": args.cutoff,
        "target": args.target,
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    with (out_dir / "config_resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved outputs to: {out_dir}")
    print(f"Final metrics - MAE: {history['val_mae'][-1]:.4f}, RMSE: {history['val_rmse'][-1]:.4f}, R²: {history['val_r2'][-1]:.3f}")


if __name__ == "__main__":
    main()
