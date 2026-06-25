# Figure pack — stats summary

Generated into `results/figures/nature/` (10 figures). Palette: Okabe-Ito (CB-safe).

## OOD headline (mean over 6 TFs)
| model | OOD AUROC |
|---|---|
| HetMoE | 0.7819 |
| DNABERT-6 | 0.7492 |
| best single | 0.7447 |
| DeepSEA | 0.6918 |
| DanQ | 0.6881 |
| static mean | 0.7276 |
| orig MoE | 0.6829 |

## HetMoE vs DNABERT-6 (OOD)
- mean ΔAUROC = +0.0326 (per-TF: BCLAF1 +0.037, CTCF +0.066, POLR2A +0.032, RBBP5 +0.012, SAP30 +0.028, STAT3 +0.021)
- standardized effect size (Cohen's d, paired across TFs) = 1.74
- selected config: 6 experts (ConvNet+DNABERT6), l2norm=True, entropy=0.001, tau=1.0

## Multi-seed robustness (OOD mean ± SD)
- HetMoE   = 0.7772 ± 0.0030  (n=5 seeds)
- DNABERT-6 = 0.7362 ± 0.0087

## OOD calibration (pooled, ECE)
- HetMoE = 0.1401
- DNABERT-6 = 0.1339
