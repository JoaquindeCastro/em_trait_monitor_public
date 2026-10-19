# Threshold removal sweep

Final PC1 drift (percent of the base norm) against the number of harmful examples left in training.

| run | mode | z | removed | harmful left | final step | final drift | step-40 drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bad_medical_fullseq_seed42 | baseline |  | 0 | 1000 | 126 | 24.29 | 13.12 |
| arm_z2_random_k34 | random delete | 2 | 34 | 966 | 122 | 23.24 | 14.11 |
| arm_z2_random_rewrite_k34 | random rewrite | 2 | 34 | 966 | 126 | 23.93 | 13.41 |
| arm_z2_top_k34 | threshold delete | 2 | 34 | 966 | 122 | 25.36 | 13.46 |
| arm_z2_rewrite_k34 | threshold rewrite | 2 | 34 | 966 | 126 | 23.83 | 13.51 |
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

Experiment 2 reference (not plotted on the same monitor): {0: 12.729, 5: 19.435, 10: 23.438, 25: 27.169, 50: 28.868, 100: 26.034}
Source: codex/harmful-fraction-sweep: extensions/harmful_fraction/deliverables/followup/onset_and_control.csv
