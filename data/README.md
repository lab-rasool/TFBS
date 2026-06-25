# `data/` conventions (non-obvious — read before using)

ChIP-seq sequence files, gzipped, one per transcription factor (TF):

```
data/train/<TF>_..._AC.seq.gz   # 3 training TFs:  ARID3A, FOXM1, GATA3
data/test/<TF>_..._B.seq.gz     # 3 in-distribution test TFs (same 3 TFs)
data/ood/<TF>_..._B.seq.gz      # 6 out-of-distribution TFs: BCLAF1, CTCF, POLR2A, RBBP5, SAP30, STAT3
```

Key points that have repeatedly been misread:

- **`_AC` / `_B` are train/test FOLDS, not labels.** The per-row label is the `Bound` column
  (column index 3).
- **`_AC` (training) files are 100% positives** (`Bound = 1`). The 1:1 negative set is
  *synthesised at load time* by dinucleotide shuffling (`ChipDataLoader.dinucshuffle`), seeded so
  every expert embeds identical sequences.
- **`_B` (test / OOD) files are curated balanced sets**: 500 real positives + 500 real negatives.
- `ChipDataLoader.load_data(shuffle=True)` is the default and adds a synthetic negative for *every*
  row — calling it on a `_B` file yields a 25/75 set (the original "legacy" behaviour). The
  **rigorous** protocol uses `load_data(shuffle=False)` to evaluate on the true 50/50 set.
- **Preprocessing:** 101 bp sequence → one-hot, zero-padded by `motiflen-1 = 23` per side →
  a `4 × 147` tensor (`seqtopad`). The literal `24` in the code is the conv kernel width, not a trim.

Canonical TF order is `tfbs.constants.TRAIN_TFS = [ARID3A, FOXM1, GATA3]`. The live loaders use
unsorted `os.listdir`; the MoE concatenates expert embeddings in that order — see
[reproduce.md](../docs/reproduce.md) for why this matters.
