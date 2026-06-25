# Statistical analysis (rigorous)

_Rigorous 50/50 protocol -- ANOVA + paired-bootstrap post-hoc (* = MoE-expert 95% CI excludes 0)_

### In-distribution

| Dataset | F(3,3996) | p | eta^2 | omega^2 | Best | MoE rank | MoE-ARID3A | MoE-FOXM1 | MoE-GATA3 |
|---|---|---|---|---|---|---|---|---|---|
| ARID3A | 23112.4 | <1e-15 | 0.946 | 0.945 | ARID3A | 2/4 | -0.0445* | +0.1199* | +0.1088* |
| FOXM1 | 18867.0 | <1e-15 | 0.934 | 0.934 | FOXM1 | 2/4 | +0.1121* | -0.0378* | +0.0988* |
| GATA3 | 23493.5 | <1e-15 | 0.946 | 0.946 | GATA3 | 2/4 | +0.1003* | +0.1075* | -0.0587* |

### Out-of-distribution

| Dataset | F(3,3996) | p | eta^2 | omega^2 | Best | MoE rank | MoE-ARID3A | MoE-FOXM1 | MoE-GATA3 |
|---|---|---|---|---|---|---|---|---|---|
| BCLAF1 | 12979.5 | <1e-15 | 0.907 | 0.907 | MoE | 1/4 | +0.0718* | +0.0015ns | +0.1272* |
| CTCF | 159955.9 | <1e-15 | 0.992 | 0.992 | ARID3A | 2/4 | -0.0020ns | +0.0148* | +0.3223* |
| POLR2A | 6478.4 | <1e-15 | 0.829 | 0.829 | FOXM1 | 2/4 | +0.0756* | -0.0126ns | +0.0702* |
| RBBP5 | 4391.5 | <1e-15 | 0.767 | 0.767 | FOXM1 | 2/4 | +0.0659* | -0.0026ns | +0.0568* |
| SAP30 | 5638.2 | <1e-15 | 0.809 | 0.809 | MoE | 1/4 | +0.0515* | +0.0043ns | +0.0872* |
| STAT3 | 2276.0 | <1e-15 | 0.631 | 0.630 | MoE | 1/4 | +0.0302* | +0.0033ns | +0.0566* |