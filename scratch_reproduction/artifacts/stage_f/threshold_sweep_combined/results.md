# Threshold removal sweep

Final PC1 drift (percent of the base norm) against the number of harmful examples left in training.

| run | mode | z | removed | harmful left | final step | final drift | step-40 drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bad_medical_fullseq_seed42 | baseline |  | 0 | 1000 | 126 | 24.26 | 13.00 |
| arm_z2_random_k34 | random delete | 2 | 34 | 966 | 122 | 23.24 | 14.11 |
| arm_z2_random_rewrite_k34 | random rewrite | 2 | 34 | 966 | 126 | 23.93 | 13.41 |
| arm_z2_top_k34 | threshold delete | 2 | 34 | 966 | 122 | 25.36 | 13.46 |
| arm_z2_rewrite_k34 | threshold rewrite | 2 | 34 | 966 | 126 | 23.83 | 13.51 |
| arm_bottom_k100 | bottom delete |  | 100 | 900 | 114 | 26.35 | 12.96 |
| arm_r1_random_k100 | random delete |  | 100 | 900 | 114 | 23.21 | 13.07 |
| arm_r2_random_k100 | random delete |  | 100 | 900 | 114 | 25.92 | 11.68 |
| arm_random_k100 | random delete |  | 100 | 900 | 114 | 23.45 | 13.13 |
| arm_top_k100 | threshold delete |  | 100 | 900 | 114 | 22.49 | 13.57 |
| arm_rewrite_k100 | threshold rewrite |  | 100 | 900 | 126 | 22.83 | 13.41 |
| arm_z1_random_k152 | random delete | 1 | 152 | 848 | 106 | 22.41 | 13.23 |
| arm_z1_random_rewrite_k152 | random rewrite | 1 | 152 | 848 | 126 | 22.76 | 13.44 |
| arm_z1_top_k152 | threshold delete | 1 | 152 | 848 | 106 | 21.11 | 11.90 |
| arm_z1_rewrite_k152 | threshold rewrite | 1 | 152 | 848 | 126 | 22.23 | 13.33 |
| arm_z0p5_random_k288 | random delete | 0.5 | 288 | 712 | 90 | 19.46 | 12.03 |
| arm_z0p5_random_rewrite_k288 | random rewrite | 0.5 | 288 | 712 | 126 | 21.74 | 13.64 |
| arm_z0p5_top_k288 | threshold delete | 0.5 | 288 | 712 | 90 | 17.37 | 11.41 |
| arm_z0p5_rewrite_k288 | threshold rewrite | 0.5 | 288 | 712 | 126 | 20.78 | 13.06 |
| arm_z0_random_k458 | random delete | 0 | 458 | 542 | 68 | 14.81 | 10.54 |
| arm_z0_random_rewrite_k458 | random rewrite | 0 | 458 | 542 | 126 | 20.51 | 12.96 |
| arm_z0_top_k458 | threshold delete | 0 | 458 | 542 | 68 | 12.36 | 8.70 |
| arm_z0_rewrite_k458 | threshold rewrite | 0 | 458 | 542 | 126 | 19.45 | 12.49 |
| mix_f25_seed42 | random rewrite |  | 750 | 250 | 126 | 18.71 | 11.91 |
| mix_f10_seed42 | random rewrite |  | 900 | 100 | 126 | 17.86 | 11.89 |
| mix_f05_seed42 | random rewrite |  | 950 | 50 | 126 | 17.46 | 11.67 |
| mix_f00_seed42 | random rewrite |  | 1000 | 0 | 126 | 17.31 | 11.59 |

Experiment 2 reference (not plotted on the same monitor): {0: 12.729, 5: 19.435, 10: 23.438, 25: 27.169, 50: 28.868, 100: 26.034}
Source: codex/harmful-fraction-sweep: extensions/harmful_fraction/deliverables/followup/onset_and_control.csv
