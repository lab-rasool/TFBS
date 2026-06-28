"""Centralised constants: TF lists, DBD-family map, OOD stratification, expert
order, and filesystem path roots. Paths are anchored to the repository root (not
the current working directory), so scripts behave identically regardless of where
they are run from.

Family-aware redesign (2026-06)
-------------------------------
The OOD split is now built around DNA-binding-domain (DBD) families and cell-line
control, drawn from the full ENCODE-690 corpus
(``huggingface.co/datasets/Lab-Rasool/ENCODE-TFBS``). Rationale:

* The previous OOD set mixed two sequence-specific TFs (CTCF, STAT3) with four
  factors that have no intrinsic motif (POLR2A = RNA Pol II; RBBP5 = COMPASS
  cofactor; SAP30 = Sin3/HDAC corepressor; BCLAF1 = RNA-binding) -- so "OOD
  generalization" was partly measured on near-unlearnable targets, and the
  pooled mean was dominated by CTCF. It also shared *no* DBD family between train
  and test, so motif knowledge could not transfer.
* The redesign trains on **seven TFs spanning six DBD families** and evaluates on
  held-out members of those families (``OOD_WITHIN_FAMILY`` -- the genuine
  motif-transfer test), entirely unseen families (``OOD_CROSS_FAMILY``), the same
  TF in unseen cell lines (``OOD_CELLLINE`` -- ENCODE-DREAM-style transfer), and a
  clearly-labelled non-motif appendix (``OOD_NONMOTIF``) reported separately.
* Cell-line confound control: training and most OOD TFs are anchored to **K562**
  so the train/OOD boundary is not confounded with cell line; the few cross-line
  cases are flagged in ``CELL_LINE_NOTE``.

The exact experiment file chosen per TF (cell line / antibody / lab) is pinned by
``experiments/data/fetch_encode.py`` into ``data/encode_manifest.json``.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path roots (repo-root anchored)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OOD_DIR = DATA_DIR / "ood"
GENOMICNEG_DIR = DATA_DIR / "train_genomicneg"   # GC-matched real-negative training files
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
ARCHIVE_DIR = ROOT / "archive"

# ---------------------------------------------------------------------------
# Training-negative mode (the "fair negatives" arm). Set TFBS_TRAIN_NEG=genomic to
# train experts on GC/repeat-matched REAL genomic negatives (built by
# experiments.analysis.data_quality build-negs) instead of dinucleotide-shuffle
# negatives -- addresses the train/eval negative-set mismatch (Tourne et al. 2026).
# Default "dinuc" reproduces the original behaviour exactly.
# ---------------------------------------------------------------------------
TRAIN_NEG_MODE = os.environ.get("TFBS_TRAIN_NEG", "dinuc").lower()
# Output-path suffix so the genomic (fair-negative) arm does not clobber the dinuc
# results: dinuc -> results/cache/seed42; genomic -> results/cache/seed42_genomic.
MODE_SUFFIX = "" if TRAIN_NEG_MODE == "dinuc" else f"_{TRAIN_NEG_MODE}"


def train_dir_and_shuffle():
    """(training-file directory, load_data shuffle flag) for the current negative
    mode. genomic: negatives are already in the file -> shuffle=False (no synthesis);
    dinuc: positives-only file -> shuffle=True (synthesize one dinuc-shuffle neg/row)."""
    if TRAIN_NEG_MODE == "genomic":
        return str(GENOMICNEG_DIR), False
    return str(TRAIN_DIR), True

# ---------------------------------------------------------------------------
# DNA-binding-domain (DBD) family map (used to justify the OOD strata and to
# build the train-vs-OOD family-similarity analysis). Families follow JASPAR /
# UniProt structural classes.
# ---------------------------------------------------------------------------
DBD_FAMILY = {
    # training factors
    "ARID3A": "ARID/Bright",
    "FOXM1": "Forkhead",
    "GATA3": "GATA-ZF",
    "JUND": "bZIP",
    "MAX": "bHLH-ZIP",
    "GABPA": "ETS",
    "SP1": "C2H2-ZF/SP",
    # within-family OOD
    "JUN": "bZIP", "FOSL1": "bZIP", "ATF3": "bZIP",
    "MYC": "bHLH-ZIP", "MXI1": "bHLH-ZIP", "USF1": "bHLH-ZIP",
    "ELF1": "ETS", "ETS1": "ETS", "SPI1": "ETS",
    "SP2": "C2H2-ZF/SP", "YY1": "C2H2-ZF", "ZNF143": "C2H2-ZF", "REST": "C2H2-ZF",
    "GATA1": "GATA-ZF", "GATA2": "GATA-ZF",
    "FOXA1": "Forkhead", "FOXA2": "Forkhead",
    # cross-family OOD (unseen families, still sequence-specific)
    "CTCF": "C2H2-ZF/CTCF", "STAT3": "STAT", "NRF1": "NRF", "HNF4A": "NuclearReceptor",
    "TCF7L2": "HMG/TCF", "ZBTB33": "C2H2-ZF/BTB",
    # non-motif / recruited (appendix only)
    "POLR2A": "None(RNAPolII)", "EP300": "None(coactivator)", "EZH2": "None(PRC2)",
    "TAF1": "None(TFIID)", "RBBP5": "None(COMPASS)", "SAP30": "None(Sin3/HDAC)",
}

# ---------------------------------------------------------------------------
# Transcription factors
# ---------------------------------------------------------------------------
# Training TFs: 7 factors spanning 6 DBD families. The original ARID3A/FOXM1/GATA3
# are retained for continuity; JUND/MAX/GABPA/SP1 add bZIP/bHLH/ETS/C2H2 coverage
# (all in K562). The MoE concatenates per-expert embeddings in this order -- keep
# it stable. NOTE: changing this list invalidates existing expert checkpoints and
# the cache; a full retrain is required (cluster).
TRAIN_TFS = ["ARID3A", "FOXM1", "GATA3", "JUND", "MAX", "GABPA", "SP1"]

# OOD strata --------------------------------------------------------------
# (1) WITHIN-FAMILY: held-out TFs sharing a DBD family with a training TF -- tests
#     genuine motif-family transfer (where sequence models are expected to work).
OOD_WITHIN_FAMILY = [
    "JUN", "FOSL1", "ATF3",        # <- JUND (bZIP)
    "MYC", "MXI1", "USF1",         # <- MAX (bHLH-ZIP)
    "ELF1", "ETS1", "SPI1",        # <- GABPA (ETS)
    "SP2", "YY1", "ZNF143", "REST",  # <- SP1 (C2H2-ZF)
    "GATA1", "GATA2",              # <- GATA3 (GATA-ZF; cross-cell-line, see note)
    "FOXA1", "FOXA2",              # <- FOXM1 (Forkhead; cross-cell-line, see note)
]
# (2) CROSS-FAMILY: entirely unseen DBD families, still sequence-specific TFs with
#     real motifs -- the harder generalization test.
OOD_CROSS_FAMILY = ["CTCF", "STAT3", "NRF1", "HNF4A", "TCF7L2", "ZBTB33"]
# (3) CELL-LINE TRANSFER: a TRAINING TF evaluated in an unseen cell line (same
#     motif, different chromatin) -- ENCODE-DREAM-style transfer. Keyed as
#     "TF@CELL" by the data tooling so it does not collide with the training entry.
OOD_CELLLINE = ["MAX@GM12878", "GABPA@HepG2", "SP1@GM12878", "CTCF@GM12878", "CTCF@HepG2"]
# (4) NON-MOTIF appendix: recruited / general factors with no intrinsic motif --
#     reported SEPARATELY (sequence-only ceiling), never in the headline mean.
OOD_NONMOTIF = ["POLR2A", "EP300", "EZH2", "TAF1", "RBBP5", "SAP30"]

# Headline OOD = the motif-bearing strata (within- + cross-family). The cell-line
# and non-motif strata are evaluated and reported separately.
OOD_TFS = OOD_WITHIN_FAMILY + OOD_CROSS_FAMILY + OOD_NONMOTIF

# ---------------------------------------------------------------------------
# Backward-compatible stratification names (consumed by figures / stats).
# LEARNABLE = motif-bearing OOD (within + cross family); INDIRECT = non-motif.
# ---------------------------------------------------------------------------
OOD_LEARNABLE = OOD_WITHIN_FAMILY + OOD_CROSS_FAMILY
OOD_INDIRECT = OOD_NONMOTIF

# Preferred cell line per TF (cell-line confound control). The fetcher picks this
# experiment when available; falls back to the most common line otherwise.
PREFERRED_CELL = {tf: "K562" for tf in set(TRAIN_TFS) | set(OOD_WITHIN_FAMILY)
                  | set(OOD_CROSS_FAMILY) | set(OOD_NONMOTIF)}
# Documented exceptions (not in K562): these are cross-cell-line OOD cases.
PREFERRED_CELL.update({
    "FOXM1": "GM12878", "GATA3": "MCF-7",
    "FOXA1": "HepG2", "FOXA2": "HepG2",
    "STAT3": "HeLa-S3", "HNF4A": "HepG2", "TCF7L2": "HeLa-S3",
    "EZH2": "GM12878",
})
CELL_LINE_NOTE = (
    "Training and most OOD TFs are anchored to K562 so the train/OOD boundary is "
    "not confounded with cell line. Exceptions (different line, flagged): FOXM1 "
    "(GM12878), GATA3 (MCF-7) and their within-family OOD (FOXA1/2 HepG2; GATA1/2 "
    "K562), STAT3/TCF7L2 (HeLa-S3), HNF4A (HepG2). The OOD_CELLLINE stratum is the "
    "deliberate same-TF/unseen-line transfer test."
)
