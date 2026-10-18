# Mixture evaluation: mix_f10_seed42 / raw_contrast_L17

- n = 1000, planted k = 100 (10%); planted mean score 9.870e-03, benign mean -1.416e-03, benign fraction positive 0.41

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.530 | 0.365 | 0.869 | 0.535 |
| final_cos | 0.450 | 0.325 | 0.843 | 0.408 |
| tracin_unweighted | 0.510 | 0.355 | 0.868 | 0.524 |
| neg_length_tokens | 0.240 | 0.195 | 0.678 | 0.176 |
| loss_first_ckpt | 0.520 | 0.395 | 0.890 | 0.521 |
| loss_last_ckpt | 0.430 | 0.330 | 0.851 | 0.416 |
| tracin_through_step30 | 0.590 | 0.390 | 0.909 | 0.621 |
| tracin_through_step40 | 0.640 | 0.400 | 0.913 | 0.627 |
| tracin_through_step50 | 0.580 | 0.365 | 0.880 | 0.573 |
| tracin_through_step70 | 0.530 | 0.350 | 0.859 | 0.530 |
| tracin_through_step100 | 0.530 | 0.360 | 0.865 | 0.533 |
| random (expected) | 0.100 | 0.100 | 0.500 | 0.100 |
