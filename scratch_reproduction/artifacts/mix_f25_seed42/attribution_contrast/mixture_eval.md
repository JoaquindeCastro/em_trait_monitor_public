# Mixture evaluation: mix_f25_seed42 / contrast

- n = 1000, planted k = 250 (25%); planted mean score 8.173e-03, benign mean 8.943e-04, benign fraction positive 0.57

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.632 | 0.438 | 0.850 | 0.702 |
| final_cos | 0.516 | 0.406 | 0.782 | 0.535 |
| tracin_unweighted | 0.592 | 0.420 | 0.828 | 0.657 |
| neg_length_tokens | 0.420 | 0.370 | 0.699 | 0.419 |
| loss_first_ckpt | 0.692 | 0.464 | 0.898 | 0.755 |
| loss_last_ckpt | 0.580 | 0.438 | 0.829 | 0.625 |
| tracin_through_step30 | 0.692 | 0.446 | 0.873 | 0.730 |
| tracin_through_step40 | 0.672 | 0.444 | 0.863 | 0.714 |
| tracin_through_step50 | 0.640 | 0.446 | 0.857 | 0.708 |
| tracin_through_step70 | 0.644 | 0.438 | 0.855 | 0.710 |
| tracin_through_step100 | 0.636 | 0.434 | 0.851 | 0.703 |
| random (expected) | 0.250 | 0.250 | 0.500 | 0.250 |
