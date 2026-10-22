#!/usr/bin/env python3
"""
download_data.py — Download C-N coupling reaction data from the Open Reaction Database (ORD).

Strategy
--------
1. Sparse-clone the ORD GitHub repo (metadata/tree only; no LFS blobs downloaded).
2. Walk the repo tree to enumerate all .pb.gz dataset files (stored as LFS pointer files).
3. Download each dataset file via the GitHub LFS Batch API (works for public repos
   without authentication).
4. Parse each downloaded .pb.gz with ord-schema; filter reactions that are C-N
   coupling (Buchwald-Hartwig / palladium-catalyzed amination).
5. Export filtered reactions to data/processed/cn_reactions_raw.csv.

Fallback
--------
If the ORD/LFS approach fails (no network, rate-limited, ord-schema not installed),
the script falls back to downloading the Ahneman 2018 Buchwald-Hartwig HTE dataset
(4,608 reactions) from the rxnfp open-source repository, which distributes it as CSV.

Usage
-----
  python scripts/download_data.py                     # Default: targeted + fallback
  python scripts/download_data.py --mode targeted     # Only download known BH datasets
  python scripts/download_data.py --mode discover     # Clone + sample all ORD datasets
  python scripts/download_data.py --mode fallback     # Direct CSV download only
  python scripts/download_data.py --max-datasets 50   # Limit ORD files to sample
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_ORD = DATA_RAW / "ord-datasets"
DATA_PROC = ROOT / "data" / "processed"
ORD_CLONE = DATA_RAW / "ord-data-meta"

ORD_REPO = "https://github.com/open-reaction-database/ord-data.git"
ORD_BASE_URL = "https://github.com/open-reaction-database/ord-data"

# ──────────────────────────────────────────────────────────────
# Fallback: Known C-N coupling dataset CSV sources
# These are publicly distributed copies of the Ahneman 2018
# Buchwald-Hartwig dataset (Science, 4608 reactions).
# ──────────────────────────────────────────────────────────────
FALLBACK_URLS = [
    # drfp repo (Probst & Reymond 2021, Paper 8) — BH dataset used in benchmarks
    "https://raw.githubusercontent.com/reymond-group/drfp/main/"
    "notebooks/data/buchwald_hartwig/FullCV_01.csv",
    # rxnfp repo (Schwaller et al.) — BH full cross-validation splits
    "https://raw.githubusercontent.com/rxn4chemistry/rxnfp/master/"
    "notebooks/FullCV_01.csv",
    # doyle-lab rxnpredict repo — original Ahneman dataset
    "https://raw.githubusercontent.com/doyle-lab-ucla/rxnpredict/main/"
    "data/Buchwald-Hartwig/BH_dataset.csv",
    # Backup: Yang et al. machine-learning-for-chemistry
    "https://raw.githubusercontent.com/yangkky/Machine-learning-for-chemistry/master/"
    "data/reaction-prediction/buchwald_hartwig.csv",
]

# ──────────────────────────────────────────────────────────────
# C-N coupling keywords used for filtering parsed ORD reactions
# ──────────────────────────────────────────────────────────────
CN_KEYWORDS = [
    "buchwald", "hartwig", "amination", "c-n coupling",
    "cn coupling", "palladium-catalyzed amination",
    "cross-coupling", "n-arylation", "ullmann",
]

PALLADIUM_SMILES_FRAGMENTS = ["[Pd]", "Pd(", "Pd2("]


# ──────────────────────────────────────────────────────────────
# Installation helpers
# ──────────────────────────────────────────────────────────────

def ensure_package(package: str, import_name: str = None) -> bool:
    """Try to import package; pip-install if missing. Returns True on success."""
    import_name = import_name or package
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"[install] Installing {package}...")
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            capture_output=True,
        )
        if ret.returncode != 0:
            print(f"[warn] Failed to install {package}: {ret.stderr.decode()[:200]}")
            return False
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False


# ──────────────────────────────────────────────────────────────
# Git / LFS helpers
# ──────────────────────────────────────────────────────────────

def git_available() -> bool:
    return bool(subprocess.run(["git", "--version"], capture_output=True).returncode == 0)


def git_lfs_available() -> bool:
    return bool(subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0)


def sparse_clone_ord_metadata():
    """
    Clone only the repo tree (no LFS blobs) to enumerate .pb.gz files.
    After this, each .pb.gz in the worktree is an LFS pointer file (~130 bytes),
    not the actual data.
    """
    if (ORD_CLONE / ".git").exists():
        print(f"[git] ORD metadata clone already exists at {ORD_CLONE}")
        return True

    ORD_CLONE.mkdir(parents=True, exist_ok=True)
    print("[git] Sparse-cloning ORD repo (metadata only, no blobs)...")

    cmds = [
        ["git", "clone", "--depth=1", "--filter=blob:none",
         "--sparse", "--no-checkout", ORD_REPO, str(ORD_CLONE)],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            print(f"[git] Error: {r.stderr.decode()[:300]}")
            return False

    # Sparse checkout: only the data/ directory (still no blobs)
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "data/"],
                   cwd=ORD_CLONE, capture_output=True)
    subprocess.run(["git", "checkout"], cwd=ORD_CLONE, capture_output=True)
    return True


def read_lfs_pointer(path: Path):
    """
    Parse an LFS pointer file.
    Returns (oid_sha256: str, size_bytes: int) or (None, None) if not an LFS pointer.
    """
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None, None

    if "oid sha256:" not in text:
        return None, None  # Actual file, not a pointer

    try:
        oid = text.split("oid sha256:")[1].split()[0].strip()
        size = int(text.split("size ")[1].split()[0].strip())
        return oid, size
    except (IndexError, ValueError):
        return None, None


def lfs_batch_download_url(oid: str, size: int, repo_url: str = ORD_REPO) -> str:
    """
    Query the GitHub LFS Batch API and return the direct download URL for an object.
    Works for public repositories without authentication.
    """
    batch_url = f"{repo_url}/info/lfs/objects/batch"
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "refs": {"name": "refs/heads/main"},
        "objects": [{"oid": oid, "size": size}],
    }).encode()

    req = urllib.request.Request(
        batch_url,
        data=payload,
        headers={
            "Content-Type": "application/vnd.git-lfs+json",
            "Accept": "application/vnd.git-lfs+json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    obj = result["objects"][0]
    if "error" in obj:
        raise RuntimeError(f"LFS batch error: {obj['error']}")
    return obj["actions"]["download"]["href"]


def download_lfs_file(pointer_path: Path, dest_path: Path) -> bool:
    """Download the actual LFS object for a pointer file to dest_path."""
    oid, size = read_lfs_pointer(pointer_path)
    if oid is None:
        # Not an LFS pointer — it's the actual file; just copy
        import shutil
        shutil.copy2(pointer_path, dest_path)
        return True

    if dest_path.exists() and dest_path.stat().st_size == size:
        print(f"  [skip] {dest_path.name} already downloaded")
        return True

    try:
        url = lfs_batch_download_url(oid, size)
        print(f"  [lfs]  Downloading {dest_path.name} ({size // 1024:,} KB)...")
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  [warn] LFS download failed for {pointer_path.name}: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# ORD protobuf parsing
# ──────────────────────────────────────────────────────────────

def is_cn_coupling_reaction(rxn) -> bool:
    """
    Heuristic filter: returns True if the reaction is likely a C-N coupling
    (Buchwald-Hartwig amination or similar palladium-catalyzed C-N bond formation).

    Checks (in order of cost, cheapest first):
      1. Reaction name / identifiers contain C-N coupling keywords
      2. Notes / procedure text contains keywords
      3. Any input component SMILES contains a palladium fragment
    """
    # 1. Identifier names
    for ident in rxn.identifiers:
        text = ident.value.lower()
        if any(kw in text for kw in CN_KEYWORDS):
            return True

    # 2. Notes
    notes_text = (rxn.notes.procedure_details or "").lower()
    if any(kw in notes_text for kw in CN_KEYWORDS):
        return True

    # 3. Check if any input has a palladium-containing SMILES (catalyst)
    has_pd = False
    has_amine = False
    has_aryl_halide = False

    for input_name, rxn_input in rxn.inputs.items():
        for component in rxn_input.components:
            for ident in component.identifiers:
                if ident.type in (2, 10):  # SMILES=2, CXSMILES=10
                    smi = ident.value
                    if any(frag in smi for frag in PALLADIUM_SMILES_FRAGMENTS):
                        has_pd = True
                    if "N" in smi and not smi.startswith("O="):
                        has_amine = True
                    if any(x in smi for x in ["Cl", "Br", "I", "F"]) and "c" in smi:
                        has_aryl_halide = True

    return has_pd and has_amine and has_aryl_halide


def extract_reaction_record(rxn) -> dict | None:
    """
    Extract a flat record from an ORD Reaction proto for a C-N coupling reaction.
    Returns a dict suitable for a DataFrame row, or None if yield is missing.

    ORD enum values used here (verified against ord_schema 0.3.99):
      CompoundIdentifier.type: SMILES=2, INCHI=3, NAME=6
      Compound.reaction_role: REACTANT=1, REAGENT=2, SOLVENT=3, CATALYST=4
      ProductMeasurement.type: YIELD=3
      ReactionIdentifier.type: REACTION_SMILES=2
    """
    record = {
        "reaction_id": rxn.reaction_id,
        "dataset_name": "",
        "yield": None,
        "aryl_halide_smiles": None,
        "amine_smiles": None,
        "catalyst_smiles": None,
        "base_smiles": None,
        "solvent_smiles": None,
        "additive_smiles": None,
        "temperature_c": None,
        "reaction_smiles": None,
    }

    # Extract yield from outcomes (YIELD measurement type = 3)
    for outcome in rxn.outcomes:
        for product in outcome.products:
            for meas in product.measurements:
                if meas.type == 3:  # YIELD
                    record["yield"] = meas.percentage.value
                    break
        if record["yield"] is not None:
            break

    if record["yield"] is None:
        return None  # Skip reactions without a recorded yield

    # Temperature (TemperatureConditions.setpoint)
    # ORD Temperature.units enum: 0=UNSPECIFIED, 1=CELSIUS, 2=FAHRENHEIT, 3=KELVIN
    try:
        t = rxn.conditions.temperature.setpoint
        if t.value != 0:
            units = t.units
            if units == 1:    # CELSIUS
                record["temperature_c"] = t.value
            elif units == 3:  # KELVIN
                record["temperature_c"] = t.value - 273.15
            elif units == 2:  # FAHRENHEIT
                record["temperature_c"] = (t.value - 32) * 5 / 9
    except Exception:
        pass

    # Reaction SMILES (ReactionIdentifier.type = REACTION_SMILES = 2)
    for ident in rxn.identifiers:
        if ident.type == 2:  # REACTION_SMILES
            record["reaction_smiles"] = ident.value
            break

    def get_smiles(comp) -> str | None:
        """Return SMILES string for a compound (type 2=SMILES, 10=CXSMILES)."""
        for ident in comp.identifiers:
            if ident.type in (2, 10):
                return ident.value
        return None

    def has_halogen(smi: str) -> bool:
        return any(x in smi for x in ["Cl", "Br", "I", "F"])

    def has_nitrogen(smi: str) -> bool:
        return "N" in smi or "n" in smi

    # Classify input components by input NAME (highest priority) then role/SMILES
    # ORD datasets often label inputs with descriptive names like "aryl halide",
    # "amine", "base", "solvent", "catalyst", "ligand", "additive", etc.
    for input_name, rxn_input in rxn.inputs.items():
        name_lower = input_name.lower()
        for comp in rxn_input.components:
            role = comp.reaction_role
            smi = get_smiles(comp)
            if smi is None:
                continue

            # ── Primary: use the input name ──────────────────────────────────
            if any(k in name_lower for k in ["aryl halide", "aryl-halide", "halide",
                                              "electrophile", "aryl bromide",
                                              "aryl chloride", "aryl iodide"]):
                if record["aryl_halide_smiles"] is None:
                    record["aryl_halide_smiles"] = smi
                continue

            if any(k in name_lower for k in ["amine", "nucleophile", "aniline",
                                              "substrate amine", "nitrogen"]):
                if record["amine_smiles"] is None:
                    record["amine_smiles"] = smi
                continue

            if any(k in name_lower for k in ["solvent"]):
                if record["solvent_smiles"] is None:
                    record["solvent_smiles"] = smi
                continue

            if any(k in name_lower for k in ["catalyst", "pd", "palladium",
                                              "metal", "ligand"]):
                if record["catalyst_smiles"] is None:
                    record["catalyst_smiles"] = smi
                continue

            if any(k in name_lower for k in ["base", "carbonate", "alkoxide",
                                              "phosphate", "amide"]):
                if record["base_smiles"] is None:
                    record["base_smiles"] = smi
                continue

            if any(k in name_lower for k in ["additive", "ligand", "reagent"]):
                if record["additive_smiles"] is None:
                    record["additive_smiles"] = smi
                continue

            # ── Fallback: classify by role then SMILES content ───────────────
            if role == 3 and record["solvent_smiles"] is None:     # SOLVENT
                record["solvent_smiles"] = smi
            elif role == 4 and record["catalyst_smiles"] is None:  # CATALYST
                record["catalyst_smiles"] = smi
            elif role == 2:  # REAGENT — base or additive
                is_inorganic = "[" in smi and not any(c.islower() for c in smi)
                if is_inorganic and record["base_smiles"] is None:
                    record["base_smiles"] = smi
                elif record["additive_smiles"] is None:
                    record["additive_smiles"] = smi
            elif role == 1:  # REACTANT — aryl halide or amine
                # Aryl halides have halogens; check halogen FIRST
                # (Kekulé SMILES uses uppercase C, so don't require lowercase c)
                if has_halogen(smi) and record["aryl_halide_smiles"] is None:
                    record["aryl_halide_smiles"] = smi
                elif has_nitrogen(smi) and record["amine_smiles"] is None:
                    record["amine_smiles"] = smi

    return record


def parse_ord_dataset(pb_gz_path: Path) -> list[dict]:
    """Parse one .pb.gz ORD dataset file; return list of C-N coupling records."""
    try:
        from ord_schema import message_helpers
        from ord_schema.proto import dataset_pb2
    except ImportError:
        print("[error] ord-schema not available. Install with: pip install ord-schema")
        return []

    records = []
    try:
        dataset = message_helpers.load_message(str(pb_gz_path), dataset_pb2.Dataset)
        n_total = len(dataset.reactions)
        n_cn = 0
        for rxn in dataset.reactions:
            if is_cn_coupling_reaction(rxn):
                rec = extract_reaction_record(rxn)
                if rec is not None:
                    records.append(rec)
                    n_cn += 1
        print(f"    {pb_gz_path.name}: {n_total} total → {n_cn} C-N coupling with yield")
    except Exception as e:
        print(f"    [warn] Could not parse {pb_gz_path.name}: {e}")

    return records


# ──────────────────────────────────────────────────────────────
# Fallback: direct CSV download (Ahneman 2018 BH dataset)
# ──────────────────────────────────────────────────────────────

def download_fallback_csv(out_path: Path) -> bool:
    """Try each FALLBACK_URL; save to out_path. Returns True on success."""
    for url in FALLBACK_URLS:
        print(f"[fallback] Trying {url[:80]}...")
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"[fallback] Saved to {out_path}")
            return True
        except Exception as e:
            print(f"[fallback] Failed: {e}")
    return False


def normalize_fallback_csv(csv_path: Path, out_path: Path):
    """
    Rename columns in the Ahneman/rxnfp CSV to match the ORD-derived format.
    The rxnfp FullCV_01.csv has columns like:
      Output, rxn, Ligand, Additive, Base, ArylHalide, ...
    We remap to our standard schema.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"[normalize] Columns: {list(df.columns)}")

    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    # Yield
    for cand in ["output", "yield", "y", "ee"]:
        if cand in col_lower:
            rename_map[col_lower[cand]] = "yield"
            break

    # SMILES columns (rxnfp stores SMILES directly)
    for src, dst in [
        ("arylhalide", "aryl_halide_smiles"),
        ("aryl_halide", "aryl_halide_smiles"),
        ("ligand", "catalyst_smiles"),
        ("base", "base_smiles"),
        ("additive", "additive_smiles"),
        ("solvent", "solvent_smiles"),
        ("rxn", "reaction_smiles"),
    ]:
        if src in col_lower:
            rename_map[col_lower[src]] = dst

    df = df.rename(columns=rename_map)

    # If yield is 0-1, scale to 0-100
    if "yield" in df.columns:
        if df["yield"].max() <= 1.01:
            df["yield"] = df["yield"] * 100

    df.to_csv(out_path, index=False)
    print(f"[normalize] Saved {len(df)} reactions to {out_path}")


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run_targeted(args):
    """
    Download ORD datasets by enumerating the sparse-cloned repo tree,
    then filtering with the C-N coupling heuristic.
    """
    if not git_available():
        print("[error] git not found. Please install git.")
        return False

    ensure_package("ord-schema", "ord_schema")

    # Step 1: Sparse clone ORD metadata
    if not sparse_clone_ord_metadata():
        print("[error] Could not clone ORD metadata.")
        return False

    # Step 2: Enumerate .pb.gz pointer files
    data_dir = ORD_CLONE / "data"
    if not data_dir.exists():
        print(f"[error] {data_dir} not found after sparse clone.")
        return False

    pointer_files = sorted(data_dir.glob("**/*.pb.gz"))
    print(f"[scan] Found {len(pointer_files)} dataset files in ORD repo.")

    if args.max_datasets:
        pointer_files = pointer_files[:args.max_datasets]
        print(f"[scan] Limiting to first {args.max_datasets} datasets.")

    # Step 3: Download via LFS batch API + parse
    all_records = []
    DATA_ORD.mkdir(parents=True, exist_ok=True)

    for pf in pointer_files:
        dest = DATA_ORD / pf.name
        ok = download_lfs_file(pf, dest)
        if ok and dest.exists():
            recs = parse_ord_dataset(dest)
            all_records.extend(recs)
            if not args.keep_raw:
                dest.unlink(missing_ok=True)

    if all_records:
        import pandas as pd
        out = DATA_PROC / "cn_reactions_raw.csv"
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_records).to_csv(out, index=False)
        print(f"\n[done] Saved {len(all_records)} C-N coupling reactions to {out}")
        return True

    print("[warn] No C-N coupling reactions found in targeted download.")
    return False


def run_fallback(args):
    """Download the Ahneman 2018 BH dataset as CSV (fallback path)."""
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    raw_csv = DATA_PROC / "ahneman_bh_raw.csv"
    out_csv = DATA_PROC / "cn_reactions_raw.csv"

    if out_csv.exists() and not args.force:
        print(f"[skip] {out_csv} already exists. Use --force to re-download.")
        return True

    ok = download_fallback_csv(raw_csv)
    if ok:
        normalize_fallback_csv(raw_csv, out_csv)
        raw_csv.unlink(missing_ok=True)
        return True

    print("[error] All fallback URLs failed. Cannot download dataset.")
    print("\nManual fallback:")
    print("  Download the Buchwald-Hartwig dataset from the Doyle lab or rxnfp repo")
    print(f"  and place it at {out_csv} with at minimum columns: yield, aryl_halide_smiles")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download C-N coupling data from the Open Reaction Database."
    )
    parser.add_argument(
        "--mode",
        choices=["targeted", "fallback", "auto"],
        default="auto",
        help="Download mode: targeted=ORD LFS, fallback=CSV only, auto=try both",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Max number of ORD dataset files to sample in targeted mode",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw .pb.gz files after parsing (large, ~MB each)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output files already exist",
    )
    args = parser.parse_args()

    out_csv = DATA_PROC / "cn_reactions_raw.csv"
    if out_csv.exists() and not args.force:
        print(f"[skip] {out_csv} already exists ({out_csv.stat().st_size // 1024} KB).")
        print("       Use --force to re-download.")
        return

    if args.mode in ("targeted", "auto"):
        success = run_targeted(args)
        if success:
            return

    if args.mode in ("fallback", "auto"):
        run_fallback(args)


if __name__ == "__main__":
    main()
