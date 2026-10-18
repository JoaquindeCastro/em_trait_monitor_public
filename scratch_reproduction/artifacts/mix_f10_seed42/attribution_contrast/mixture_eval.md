# Mixture evaluation: mix_f10_seed42 / contrast

- n = 1000, planted k = 100 (10%); planted mean score 9.481e-03, benign mean 1.478e-03, benign fraction positive 0.63

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.540 | 0.375 | 0.896 | 0.530 |
| final_cos | 0.340 | 0.280 | 0.804 | 0.282 |
| tracin_unweighted | 0.440 | 0.355 | 0.874 | 0.475 |
| neg_length_tokens | 0.240 | 0.195 | 0.678 | 0.176 |
| loss_first_ckpt | 0.520 | 0.395 | 0.890 | 0.521 |
| loss_last_ckpt | 0.430 | 0.330 | 0.851 | 0.416 |
| tracin_through_step30 | 0.510 | 0.335 | 0.864 | 0.498 |
| tracin_through_step40 | 0.550 | 0.370 | 0.882 | 0.534 |
| tracin_through_step50 | 0.510 | 0.370 | 0.885 | 0.527 |
| tracin_through_step70 | 0.510 | 0.370 | 0.893 | 0.536 |
| tracin_through_step100 | 0.530 | 0.375 | 0.897 | 0.536 |
| random (expected) | 0.100 | 0.100 | 0.500 | 0.100 |
