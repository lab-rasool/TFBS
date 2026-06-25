# Heterogeneous Embedding-Gated MoE — results report

### Headline: OOD AUC (balanced 50/50, B=1000 bootstrap)

| Model | OOD mean AUC | In-dist AUC |
|---|---|---|
| HetMoE (ours) | 0.7819 | 0.8334 |
| DNABERT-6 | 0.7467 | -- |
| Static-mean (same zoo) | 0.7276 | -- |
| Best-single expert | 0.7447 | -- |
| DeepSEA (baseline) | 0.6918 | -- |
| DanQ (baseline) | 0.6881 | -- |
| Original MoE (paper) | 0.6829 | -- |

Stratified OOD HetMoE: learnable (CTCF,STAT3) = **0.8364**, indirect = 0.7546.  OOD calibration: ECE = 0.154, Brier = 0.216.

### HetMoE vs DNABERT-6 (paired bootstrap on identical resamples)

TOST non-inferiority margin = 0.01.  SUP = 95% CI excludes 0 (superior); NI = non-inferior (TOST).

| OOD TF | ΔAUC (HetMoE−DNABERT) | 95% CI | p | verdict |
|---|---|---|---|---|
| BCLAF1 | +0.0428 | [+0.0287, +0.0577] | 0.000 | SUP |
| CTCF | +0.0740 | [+0.0584, +0.0905] | 0.000 | SUP |
| POLR2A | +0.0197 | [+0.0034, +0.0362] | 0.014 | SUP |
| RBBP5 | +0.0256 | [+0.0098, +0.0407] | 0.002 | SUP |
| SAP30 | +0.0237 | [+0.0086, +0.0380] | 0.000 | SUP |
| STAT3 | +0.0261 | [+0.0144, +0.0373] | 0.000 | SUP |

**Superior on 6/6, non-inferior on 6/6 OOD TFs.**

### Ablation: backbone set / N_e / gate knobs (selection on in-dist only)

| Config | N_e | OOD AUC | OOD oracle | in-dist | gate entropy |
|---|---|---|---|---|---|
| convnet_dnabert6_6 | 6 | 0.7819 | 0.7819 | 0.8334 | 1.38 |
| full12_tau2 | 12 | 0.7744 | 0.7843 | 0.8325 | 2.32 |
| full12 | 12 | 0.7743 | 0.7843 | 0.8326 | 2.19 |
| full12_ent1e-2 | 12 | 0.7736 | 0.7843 | 0.8290 | 2.34 |
| grouped4 | 4 | 0.7721 | 0.7567 | 0.8292 | 1.25 |
| dnabert6_only_3 | 3 | 0.7690 | 0.7662 | 0.8232 | 0.91 |
| full12_noknobs | 12 | 0.7688 | 0.7843 | 0.8280 | 1.34 |
| cnn_only_9 | 9 | 0.7392 | 0.7333 | 0.7948 | 1.99 |
| convnet_only_3 | 3 | 0.6847 | 0.6844 | 0.7186 | 0.83 |