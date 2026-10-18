# Mixture evaluation: mix_f10_seed42 / raw_contrast_L18

- n = 1000, planted k = 100 (10%); planted mean score 8.197e-03, benign mean -1.117e-03, benign fraction positive 0.41

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.540 | 0.380 | 0.888 | 0.578 |
| final_cos | 0.490 | 0.350 | 0.870 | 0.466 |
| tracin_unweighted | 0.560 | 0.380 | 0.889 | 0.580 |
| neg_length_tokens | 0.240 | 0.195 | 0.678 | 0.176 |
| loss_first_ckpt | 0.520 | 0.395 | 0.890 | 0.521 |
| loss_last_ckpt | 0.430 | 0.330 | 0.851 | 0.416 |
| tracin_through_step30 | 0.660 | 0.440 | 0.944 | 0.723 |
| tracin_through_step40 | 0.660 | 0.430 | 0.937 | 0.696 |
| tracin_through_step50 | 0.610 | 0.405 | 0.897 | 0.613 |
| tracin_through_step70 | 0.550 | 0.385 | 0.877 | 0.569 |
| tracin_through_step100 | 0.540 | 0.385 | 0.884 | 0.573 |
| random (expected) | 0.100 | 0.100 | 0.500 | 0.100 |
