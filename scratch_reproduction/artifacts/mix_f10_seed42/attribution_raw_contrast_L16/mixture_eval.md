# Mixture evaluation: mix_f10_seed42 / raw_contrast_L16

- n = 1000, planted k = 100 (10%); planted mean score 9.372e-03, benign mean -1.280e-03, benign fraction positive 0.44

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.440 | 0.315 | 0.820 | 0.451 |
| final_cos | 0.300 | 0.265 | 0.764 | 0.283 |
| tracin_unweighted | 0.410 | 0.300 | 0.802 | 0.417 |
| neg_length_tokens | 0.240 | 0.195 | 0.678 | 0.176 |
| loss_first_ckpt | 0.520 | 0.395 | 0.890 | 0.521 |
| loss_last_ckpt | 0.430 | 0.330 | 0.851 | 0.416 |
| tracin_through_step30 | 0.550 | 0.390 | 0.898 | 0.604 |
| tracin_through_step40 | 0.600 | 0.390 | 0.897 | 0.594 |
| tracin_through_step50 | 0.560 | 0.340 | 0.858 | 0.533 |
| tracin_through_step70 | 0.450 | 0.330 | 0.829 | 0.475 |
| tracin_through_step100 | 0.430 | 0.310 | 0.820 | 0.453 |
| random (expected) | 0.100 | 0.100 | 0.500 | 0.100 |
