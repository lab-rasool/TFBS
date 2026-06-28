# Train-vs-OOD k-mer (4-mer) profile similarity

Cosine similarity of each OOD TF's positive-sequence 4-mer profile to each training TF. Higher = sequence-closer (a proxy for motif-family relatedness).

## within_family  (mean max-sim-to-train = 0.959)
| OOD TF | family | max sim | closest train | same family? |
|---|---|---|---|---|
| JUN | bZIP | 0.984 | JUND | yes |
| FOSL1 | bZIP | 0.975 | JUND | yes |
| ATF3 | bZIP | 0.942 | MAX | no |
| MYC | bHLH-ZIP | 0.989 | MAX | yes |
| MXI1 | bHLH-ZIP | 0.970 | MAX | yes |
| USF1 | bHLH-ZIP | 0.972 | MAX | yes |
| ELF1 | ETS | 0.979 | GABPA | yes |
| ETS1 | ETS | 0.967 | MAX | no |
| SPI1 | ETS | 0.932 | FOXM1 | no |
| SP2 | C2H2-ZF/SP | 0.984 | SP1 | yes |
| YY1 | C2H2-ZF | 0.952 | SP1 | no |
| ZNF143 | C2H2-ZF | 0.938 | SP1 | no |
| REST | C2H2-ZF | 0.906 | SP1 | no |
| GATA1 | GATA-ZF | 0.912 | JUND | no |
| GATA2 | GATA-ZF | 0.963 | GATA3 | yes |
| FOXA1 | Forkhead | 0.973 | ARID3A | no |
| FOXA2 | Forkhead | 0.974 | ARID3A | no |

## cross_family  (mean max-sim-to-train = 0.933)
| OOD TF | family | max sim | closest train | same family? |
|---|---|---|---|---|
| CTCF | C2H2-ZF/CTCF | 0.920 | SP1 | no |
| STAT3 | STAT | 0.967 | FOXM1 | no |
| NRF1 | NRF | 0.917 | MAX | no |
| HNF4A | NuclearReceptor | 0.937 | FOXM1 | no |
| TCF7L2 | HMG/TCF | 0.924 | FOXM1 | no |
| ZBTB33 | C2H2-ZF/BTB | 0.930 | SP1 | no |

## non_motif  (mean max-sim-to-train = 0.958)
| OOD TF | family | max sim | closest train | same family? |
|---|---|---|---|---|
| POLR2A | None(RNAPolII) | 0.952 | MAX | no |
| EP300 | None(coactivator) | 0.956 | GATA3 | no |
| EZH2 | None(PRC2) | 0.969 | FOXM1 | no |
| TAF1 | None(TFIID) | 0.953 | GABPA | no |
| RBBP5 | None(COMPASS) | 0.956 | MAX | no |
| SAP30 | None(Sin3/HDAC) | 0.966 | MAX | no |

## Summary
- **within_family**: mean max-sim 0.959; closest-train-TF-is-same-family for 8/17
- **cross_family**: mean max-sim 0.933; closest-train-TF-is-same-family for 0/6
- **non_motif**: mean max-sim 0.958; closest-train-TF-is-same-family for 0/6