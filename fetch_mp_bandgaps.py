#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from mp_api.client import MPRester
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a pilot subset (~100) of Materials Project crystals with band gap labels "
            "and save to JSON, Parquet, and CIF files."
        )
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("MP_API_KEY"),
        help="Materials Project API key. Defaults to MP_API_KEY env var if set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of materials to fetch (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. If not provided, will create ./data/mp_subset_<YYYYmmdd_HHMMSS>"
        ),
    )
    parser.add_argument(
        "--only-stable",
        action="store_true",
        help="If set, restrict to thermodynamically stable materials (energy_above_hull == 0).",
    )
    parser.add_argument(
        "--elements",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of element symbols to filter (e.g., 'Si,O'). "
            "If omitted, fetches from the whole database."
        ),
    )
    return parser.parse_args()


def ensure_output_dir(base_dir: str | None) -> Path:
    if base_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"data/mp_subset_{timestamp}"
    out = Path(base_dir).expanduser().resolve()
    (out / "cifs").mkdir(parents=True, exist_ok=True)
    return out


def fetch_materials(api_key: str | None, limit: int, only_stable: bool, elements: List[str] | None) -> List[Any]:
    if not api_key:
        raise ValueError(
            "No API key provided. Pass --api-key or set MP_API_KEY environment variable."
        )

    fields = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "structure",
        "energy_above_hull",
        "density",
        "volume",
        "nsites",
        "symmetry",
    ]

    # Build filters
    band_gap = (0, None)  # require non-negative band gap; will include metals as 0.0
    is_stable = True if only_stable else None

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=band_gap,
            is_stable=is_stable,
            elements=elements if elements else None,
            fields=fields,
            num_chunks=1,
            chunk_size=limit,
        )

    # docs is a list of SummaryDoc objects
    return docs[:limit]


def serialize_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for d in docs:
        # Some fields may be None depending on entry
        symmetry_info = getattr(d, "symmetry", None) or {}
        serialized.append(
            {
                "material_id": str(d.material_id),
                "formula_pretty": getattr(d, "formula_pretty", None),
                "band_gap": getattr(d, "band_gap", None),
                "energy_above_hull": getattr(d, "energy_above_hull", None),
                "density": getattr(d, "density", None),
                "volume": getattr(d, "volume", None),
                "num_sites": getattr(d, "num_sites", getattr(d, "nsites", None)),
                "spacegroup": getattr(symmetry_info, "symbol", None)
                if hasattr(symmetry_info, "symbol")
                else symmetry_info.get("symbol") if isinstance(symmetry_info, dict) else None,
            }
        )
    return serialized


def save_json(data: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_parquet(data: List[Dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(data)
    df.to_parquet(out_path, index=False)


def save_cifs(docs: List[Any], out_dir: Path) -> None:
    cifs_dir = out_dir / "cifs"
    for d in tqdm(docs, desc="Writing CIFs"):
        try:
            structure = d.structure
            mpid = str(d.material_id)
            cif_path = cifs_dir / f"{mpid}.cif"
            structure.to(fmt="cif", filename=str(cif_path))
        except Exception as e:
            # Continue even if some structures cannot be written
            print(f"Warning: failed to write CIF for {getattr(d, 'material_id', 'unknown')}: {e}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_dir)

    print(f"Output directory: {out_dir}")

    if args.elements:
        elements = [e.strip() for e in args.elements.split(",") if e.strip()]
    else:
        elements = None

    docs = fetch_materials(
        api_key=args.api_key,
        limit=args.limit,
        only_stable=args.only_stable,
        elements=elements,
    )
    print(f"Fetched {len(docs)} materials")

    # Save tabular data
    records = serialize_docs(docs)
    json_path = out_dir / "materials.json"
    parquet_path = out_dir / "materials.parquet"
    save_json(records, json_path)
    save_parquet(records, parquet_path)
    print(f"Saved JSON: {json_path}")
    print(f"Saved Parquet: {parquet_path}")

    # Save structures as CIFs
    save_cifs(docs, out_dir)
    print(f"Saved CIFs to: {out_dir / 'cifs'}")


if __name__ == "__main__":
    main() 