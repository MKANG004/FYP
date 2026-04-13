# Materials Project band gap prediction (ML baselines)

Supervised regression models predict the Materials Project **band gap** (`band_gap`, eV) from crystal structure fields in `MP_dataset.csv` (lattice parameters + `sites` JSON). Four baselines are provided:

- Cartesian coordinate model — `train_cartesian_fast.py`
- Radius-graph message passing — `train_graph_pure.py`
- SchNet-style 3D model — `train_schnet_csv.py`
- CGCNN-style crystal graph model — `train_cgcnn_csv.py`

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch
