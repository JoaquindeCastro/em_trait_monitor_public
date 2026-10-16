# Mixture evaluation: mix_f10_seed42 / qwen_pc1

- n = 1000, planted k = 100 (10%); planted mean score 8.938e-03, benign mean 2.304e-03, benign fraction positive 0.66

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.390 | 0.270 | 0.789 | 0.333 |
| final_cos | 0.250 | 0.210 | 0.717 | 0.189 |
| tracin_unweighted | 0.370 | 0.260 | 0.770 | 0.307 |
| neg_length_tokens | 0.240 | 0.195 | 0.678 | 0.176 |
| loss_first_ckpt | 0.520 | 0.395 | 0.890 | 0.521 |
| loss_last_ckpt | 0.430 | 0.330 | 0.851 | 0.416 |
| tracin_through_step30 | 0.280 | 0.200 | 0.668 | 0.268 |
| tracin_through_step40 | 0.360 | 0.250 | 0.756 | 0.331 |
| tracin_through_step50 | 0.330 | 0.250 | 0.771 | 0.327 |
| tracin_through_step70 | 0.370 | 0.275 | 0.791 | 0.339 |
| tracin_through_step100 | 0.380 | 0.270 | 0.791 | 0.333 |
| random (expected) | 0.100 | 0.100 | 0.500 | 0.100 |
