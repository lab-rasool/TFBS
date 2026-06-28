# Multi-seed robustness (HetMoE vs DNABERT-6, OOD)

> **SUPERSEDED — historical.** This is the earlier **3-training-factor, 6-OOD, dinucleotide-shuffle**
> run (5 seeds). The paper's headline is the **7-factor, family-stratified, GC-matched genomic-negative**
> study: see `results/hetmoe/genomic_multiseed_summary.txt` (HetMoE 0.821±0.005 vs DNABERT-6
> 0.799±0.008 over seeds 0/1/42) and `docs/RESULTS_HETMOE.md`. Kept only as a record of the dinuc arm.

Seeds: [0, 1, 2, 3, 42]  (n=5)

| metric | mean | std | min | max |
|---|---|---|---|---|
| HetMoE OOD AUC | 0.7772 | 0.0030 | 0.7737 | 0.7819 |
| DNABERT OOD AUC | 0.7362 | 0.0087 | 0.7247 | 0.7467 |
| margin (HetMoE-DNABERT) | +0.0410 | 0.0086 | +0.0310 | +0.0530 |
| #OOD-TFs superior (of 6) | 4.0 | 1.4 | 2 | 6 |

Selected config per seed: s0:ConvNet+DNABERT6+DanQ+DeepSEA, s1:ConvNet+DNABERT6+DanQ+DeepSEA, s2:ConvNet+DNABERT6+DanQ+DeepSEA, s3:ConvNet+DNABERT6+DanQ+DeepSEA, s42:ConvNet+DNABERT6

### Per-OOD-TF mean margin (HetMoE - DNABERT) across seeds

| OOD TF | mean ΔAUC | std | seeds superior |
|---|---|---|---|
| BCLAF1 | +0.0413 | 0.0092 | 5/5 |
| CTCF | +0.1304 | 0.0376 | 5/5 |
| POLR2A | +0.0254 | 0.0116 | 5/5 |
| RBBP5 | +0.0146 | 0.0093 | 5/5 |
| SAP30 | +0.0070 | 0.0104 | 4/5 |
| STAT3 | +0.0288 | 0.0102 | 5/5 |