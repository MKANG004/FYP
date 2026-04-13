# Materials Project band gap prediction (ML baselines)

This repository trains supervised regression models to predict the Materials Project **band gap** (`band_gap`, in **eV**) from crystal structure information stored in `MP_dataset.csv` (lattice parameters + `sites` JSON).

Four baselines are included:

| Model | Script |
|------|--------|
| Cartesian coordinate model | `train_cartesian_fast.py` |
| Radius-graph message passing | `train_graph_pure.py` |
| SchNet-style 3D model | `train_schnet_csv.py` |
| CGCNN-style crystal graph model | `train_cgcnn_csv.py` |

---

## Requirements

- Python **3.10+** recommended  
- GPU optional (CPU works, but training is slower)

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install torch
