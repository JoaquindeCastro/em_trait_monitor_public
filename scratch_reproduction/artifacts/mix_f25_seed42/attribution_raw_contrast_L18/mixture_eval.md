# Mixture evaluation: mix_f25_seed42 / raw_contrast_L18

- n = 1000, planted k = 250 (25%); planted mean score 6.772e-03, benign mean -1.622e-03, benign fraction positive 0.41

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.604 | 0.426 | 0.827 | 0.665 |
| final_cos | 0.520 | 0.400 | 0.783 | 0.544 |
| tracin_unweighted | 0.568 | 0.416 | 0.807 | 0.628 |
| neg_length_tokens | 0.420 | 0.370 | 0.699 | 0.419 |
| loss_first_ckpt | 0.692 | 0.464 | 0.898 | 0.755 |
| loss_last_ckpt | 0.580 | 0.438 | 0.829 | 0.625 |
| tracin_through_step30 | 0.796 | 0.494 | 0.957 | 0.885 |
| tracin_through_step40 | 0.784 | 0.486 | 0.939 | 0.849 |
| tracin_through_step50 | 0.712 | 0.462 | 0.891 | 0.782 |
| tracin_through_step70 | 0.648 | 0.424 | 0.846 | 0.707 |
| tracin_through_step100 | 0.612 | 0.426 | 0.827 | 0.667 |
| random (expected) | 0.250 | 0.250 | 0.500 | 0.250 |
