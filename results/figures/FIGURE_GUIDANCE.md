# Figure redesign guidance (Reviewer 2, Comment 8)

The reviewer flagged Figures 1, 5 and 8 as crowded / low-resolution. Figures 3, 4,
6 and 7 (bar charts and ANOVA CI boxes) are regenerated directly from data at >=300
dpi with a colorblind-safe palette, larger fonts and direct numeric labels.

## Figure 1 - Visual abstract (hand-made; editable source not in repo)
`fig1_visual_abstract_mockup.pdf` is a decluttered layout proposal. For the final
version: (a) keep a single left-to-right flow (data -> 3 experts -> embeddings +
gating -> MoE -> evaluation/attribution); (b) >=12 pt sans-serif labels; (c) drop
decorative artwork; (d) one accent color per stage; (e) export vector PDF.
ACTION NEEDED: drop the editable source (.ai/.pptx/.svg) into paper/figures/ for a
production pass.

## Figure 5 - OOD ROC panel
`fig5_roc_out_of_distribution.pdf`: 2x3 grid, one TF per subplot, 4 model curves,
larger fonts, AUC in the legend, MoE dashed. Much less crowded than a single packed
panel.

## Figure 8 - ShiftSmooth illustration (hand-made; editable source not in repo)
`fig8_shiftsmooth_mockup.pdf` is a schematic proposal (shift -> gradient -> align +
average). ACTION NEEDED: provide the editable source for a production pass.
