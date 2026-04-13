#!/usr/bin/env python3
"""
CGCNN-style model for CSV data (MP_dataset.csv).
Crystal Graph Convolutional Neural Network with edge distance features.
Pure PyTorch implementation (no PyG required).
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

# Element to atomic number mapping
ELEMENT_MAP = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
    'Es': 99, 'Fm': 100
}


def fractional_to_cartesian(frac_coords, a, b, c, alpha, beta, gamma):
    """Convert fractional coordinates to Cartesian using lattice parameters."""
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)
    
    v1 = np.array([a, 0, 0])
    v2 = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    v3_x = c * np.cos(beta_rad)
    v3_y = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    v3_z_sq = c**2 - v3_x**2 - v3_y**2
    v3_z = np.sqrt(max(0, v3_z_sq))
    v3 = np.array([v3_x, v3_y, v3_z])
    
    lattice_matrix = np.array([v1, v2, v3]).T
    cart_coords = np.dot(lattice_matrix, frac_coords.T).T
    return cart_coords


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


class GraphSample:
    def __init__(self, z, pos, edge_index, edge_attr, y):
        self.z = z              # [N] atomic numbers
        self.pos = pos          # [N,3] positions  
        self.edge_index = edge_index  # [2, E] edges
        self.edge_attr = edge_attr    # [E, num_edge_feat] edge features (distances)
        self.y = y              # [1] target


def gaussian_expansion(distances, num_gaussians=40, cutoff=5.0):
    """Expand distances using Gaussian basis functions."""
    mu = np.linspace(0, cutoff, num_gaussians)
    sigma = cutoff / num_gaussians
    expanded = np.exp(-0.5 * ((distances[:, np.newaxis] - mu) / sigma) ** 2)
    return expanded


def build_graph(z, pos, cutoff=5.0, max_neighbors=12, num_edge_gaussians=40):
    """Build graph with edge features from atomic positions."""
    N = len(z)
    if N <= 1:
        return None
    
    # Compute all pairwise distances
    pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    diffs = pos_np[:, np.newaxis, :] - pos_np[np.newaxis, :, :]  # [N, N, 3]
    dists = np.linalg.norm(diffs, axis=-1)  # [N, N]
    
    # Build edge list with distance features
    edge_src = []
    edge_dst = []
    edge_dists = []
    
    for i in range(N):
        # Get neighbors within cutoff (excluding self)
        neighbors = []
        for j in range(N):
            if i != j and dists[i, j] <= cutoff:
                neighbors.append((j, dists[i, j]))
        
        # Sort by distance and take top max_neighbors
        neighbors.sort(key=lambda x: x[1])
        neighbors = neighbors[:max_neighbors]
        
        for j, d in neighbors:
            edge_src.append(i)
            edge_dst.append(j)
            edge_dists.append(d)
    
    if len(edge_src) == 0:
        return None
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_dists = np.array(edge_dists)
    edge_attr = gaussian_expansion(edge_dists, num_gaussians=num_edge_gaussians, cutoff=cutoff)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    return edge_index, edge_attr


class CGConvLayer(nn.Module):
    """Crystal Graph Convolution layer with edge features."""
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.node_dim = node_dim
        # Edge network: processes concatenated [node_i, node_j, edge_ij]
        # Using simple MLP without BatchNorm (BatchNorm fails with single-sample training)
        self.edge_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.Sigmoid()
        )
        self.node_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.Softplus()
        )
        
    def forward(self, h, edge_index, edge_attr):
        """
        h: [N, node_dim] node features
        edge_index: [2, E] edge indices
        edge_attr: [E, edge_dim] edge features
        """
        src, dst = edge_index
        N = h.shape[0]
        
        # Get source and destination node features
        h_src = h[src]  # [E, node_dim]
        h_dst = h[dst]  # [E, node_dim]
        
        # Concatenate node and edge features
        concat = torch.cat([h_src, h_dst, edge_attr], dim=-1)  # [E, 2*node_dim + edge_dim]
        
        # Compute edge gates and messages
        gates = self.edge_net(concat)  # [E, node_dim]
        messages = self.node_net(concat)  # [E, node_dim]
        gated_messages = gates * messages  # [E, node_dim]
        
        # Aggregate messages to nodes
        h_new = torch.zeros_like(h)
        h_new.index_add_(0, dst, gated_messages)
        
        # Normalize by number of incoming edges
        edge_count = torch.zeros(N, device=h.device)
        edge_count.index_add_(0, dst, torch.ones(len(dst), device=h.device))
        edge_count = edge_count.clamp(min=1).unsqueeze(-1)
        h_new = h_new / edge_count
        
        return h + h_new  # Residual connection


class CGCNN(nn.Module):
    """Crystal Graph Convolutional Neural Network."""
    def __init__(self, hidden_dim=64, edge_dim=40, num_conv_layers=3):
        super().__init__()
        self.embed = nn.Embedding(120, hidden_dim)
        self.conv_layers = nn.ModuleList([
            CGConvLayer(hidden_dim, edge_dim) for _ in range(num_conv_layers)
        ])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, z, edge_index, edge_attr):
        """
        z: [N] atomic numbers
        edge_index: [2, E] edge indices
        edge_attr: [E, edge_dim] edge features
        """
        h = self.embed(z)  # [N, hidden_dim]
        
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_attr)
        
        # Global mean pooling
        hg = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        out = self.output(hg)
        return out


def load_dataset_from_csv(csv_path: Path, cutoff: float, target_col: str = "band_gap", 
                          max_samples: int = None, max_neighbors: int = 12) -> List[GraphSample]:
    """Load dataset from CSV with JSON sites column."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[target_col, "sites"]).reset_index(drop=True)
    
    if max_samples:
        df = df.head(max_samples)
    
    print(f"Found {len(df)} potential samples")
    samples = []
    skipped = 0
    
    for idx, row in df.iterrows():
        if (idx + 1) % 5000 == 0:
            print(f"Processing {idx + 1}/{len(df)}... ({len(samples)} loaded so far)")
        
        try:
            a = float(row.get("a", 0))
            b = float(row.get("b", 0))
            c = float(row.get("c", 0))
            alpha = float(row.get("alpha", 90.0))
            beta = float(row.get("beta", 90.0))
            gamma = float(row.get("gamma", 90.0))
            
            if a <= 0 or b <= 0 or c <= 0:
                skipped += 1
                continue
            
            sites_str = row["sites"]
            sites = json.loads(sites_str)
            
            if not sites:
                skipped += 1
                continue
            
            atomic_numbers = []
            frac_coords = []
            
            for site in sites:
                element = site.get("element", site.get("label", ""))
                if element not in ELEMENT_MAP:
                    continue
                atomic_numbers.append(ELEMENT_MAP[element])
                fx = float(site.get("f_x", site.get("abc", [0, 0, 0])[0]))
                fy = float(site.get("f_y", site.get("abc", [0, 0, 0])[1]))
                fz = float(site.get("f_z", site.get("abc", [0, 0, 0])[2]))
                frac_coords.append([fx, fy, fz])
            
            if len(atomic_numbers) < 2:
                skipped += 1
                continue
            
            frac_coords = np.array(frac_coords)
            cart_coords = fractional_to_cartesian(frac_coords, a, b, c, alpha, beta, gamma)
            
            z = torch.tensor(atomic_numbers, dtype=torch.long)
            pos = torch.tensor(cart_coords, dtype=torch.float32)
            
            # Build graph with edge features
            result = build_graph(z, pos, cutoff=cutoff, max_neighbors=max_neighbors)
            if result is None:
                skipped += 1
                continue
            
            edge_index, edge_attr = result
            y = torch.tensor([float(row[target_col])], dtype=torch.float32)
            
            samples.append(GraphSample(z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y))
            
        except Exception:
            skipped += 1
            continue
    
    print(f"Successfully loaded {len(samples)} samples (skipped {skipped})")
    return samples


def split_dataset(samples, val_frac: float, seed: int):
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
    ap = argparse.ArgumentParser(description="CGCNN training on CSV data")
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--target", type=str, default="band_gap", help="Target column")
    ap.add_argument("--max-samples", type=int, default=None, help="Max samples to load")
    ap.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff")
    ap.add_argument("--max-neighbors", type=int, default=12, help="Max neighbors per atom")
    ap.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4, lower to prevent divergence)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--output", type=str, default=None, help="Output directory")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    print(f"Using device: {device}")

    out_dir = Path(args.output or (Path("runs") / f"cgcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset_from_csv(
        Path(args.csv), 
        cutoff=args.cutoff, 
        target_col=args.target, 
        max_samples=args.max_samples,
        max_neighbors=args.max_neighbors
    )
    train, val = split_dataset(samples, val_frac=0.2, seed=args.seed)
    print(f"Train: {len(train)}, Val: {len(val)}")

    model = CGCNN(hidden_dim=64, edge_dim=40, num_conv_layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Save config
    cfg = {
        "csv": args.csv,
        "target": args.target,
        "max_samples": args.max_samples,
        "cutoff": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed
    }
    with (out_dir / "config_resolved.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    history = {"epoch": [], "val_mae": [], "val_rmse": [], "val_r2": []}
    best = float("inf")

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for s in train:
            z = s.z.to(device)
            edge_index = s.edge_index.to(device)
            edge_attr = s.edge_attr.to(device)
            y = s.y.to(device)
            
            pred = model(z, edge_index, edge_attr)
            loss = criterion(pred.view_as(y), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_loss += loss.item()

        # Validation: compute metrics incrementally to avoid OOM on large val set
        model.eval()
        total_ae = 0.0
        total_sq_err = 0.0
        sum_y = 0.0
        sum_y_sq = 0.0
        n_val = 0
        with torch.no_grad():
            for s in val:
                z = s.z.to(device)
                edge_index = s.edge_index.to(device)
                edge_attr = s.edge_attr.to(device)
                y = s.y.to(device)
                pred = model(z, edge_index, edge_attr)
                y_np = y.cpu().view(-1)
                p_np = pred.cpu().view(-1)
                total_ae += torch.sum(torch.abs(y_np - p_np)).item()
                total_sq_err += torch.sum((y_np - p_np) ** 2).item()
                sum_y += torch.sum(y_np).item()
                sum_y_sq += torch.sum(y_np ** 2).item()
                n_val += y_np.numel()
        mean_y = sum_y / n_val if n_val else 0.0
        ss_tot = sum_y_sq - n_val * mean_y * mean_y
        ss_tot = max(ss_tot, 1e-12)
        vmae = total_ae / n_val if n_val else 0.0
        vrmse = (total_sq_err / n_val) ** 0.5 if n_val else 0.0
        vr2 = 1.0 - (total_sq_err / ss_tot)
        
        history["epoch"].append(epoch)
        history["val_mae"].append(vmae)
        history["val_rmse"].append(vrmse)
        history["val_r2"].append(vr2)
        
        train_loss_avg = train_loss / len(train)
        print(f"Epoch {epoch:3d} | loss: {train_loss_avg:.4f} | MAE: {vmae:.4f} | RMSE: {vrmse:.4f} | R2: {vr2:.3f}")

        if vmae < best:
            best = vmae
            torch.save({"model_state": model.state_dict()}, out_dir / "model.pt")
        
        # Save metrics after EACH epoch (so progress isn't lost if stopped)
        with (out_dir / "metrics.json").open("w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[SUCCESS] Training complete!")
    print(f"[SUCCESS] Saved to: {out_dir}")
    print(f"[SUCCESS] Final MAE: {vmae:.4f}, RMSE: {vrmse:.4f}, R2: {vr2:.3f}")


if __name__ == "__main__":
    main()
