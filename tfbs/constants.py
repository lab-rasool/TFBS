"""Centralised constants: TF lists, OOD stratification, expert order, and
filesystem path roots. Paths are anchored to the repository root (not the current
working directory), so scripts behave identically regardless of where they are run
from. Previously these were duplicated across hetmoe.py / stats.py / baselines.py /
aggregate_seeds.py / make_hetmoe_report.py.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Path roots (repo-root anchored)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OOD_DIR = DATA_DIR / "ood"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
ARCHIVE_DIR = ROOT / "archive"

# ---------------------------------------------------------------------------
# Transcription factors
# ---------------------------------------------------------------------------
# Canonical training-TF order. The live data loaders use unsorted os.listdir;
# downstream HetMoE/stats/reporting code pins this order explicitly. The MoE
# concatenates per-expert embeddings in this order, so keep it stable.
TRAIN_TFS = ["ARID3A", "FOXM1", "GATA3"]
OOD_TFS = ["BCLAF1", "CTCF", "POLR2A", "RBBP5", "SAP30", "STAT3"]

# OOD stratification: sequence-specific factors a sequence model can learn vs
# indirect/recruited factors with weak-or-no intrinsic motif (sequence-only ceiling).
OOD_LEARNABLE = ["CTCF", "STAT3"]
OOD_INDIRECT = ["BCLAF1", "POLR2A", "RBBP5", "SAP30"]
