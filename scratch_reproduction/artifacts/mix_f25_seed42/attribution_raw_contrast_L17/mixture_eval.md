# Mixture evaluation: mix_f25_seed42 / raw_contrast_L17

- n = 1000, planted k = 250 (25%); planted mean score 7.864e-03, benign mean -1.953e-03, benign fraction positive 0.40

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.568 | 0.408 | 0.802 | 0.625 |
| final_cos | 0.476 | 0.372 | 0.746 | 0.495 |
| tracin_unweighted | 0.536 | 0.396 | 0.777 | 0.585 |
| neg_length_tokens | 0.420 | 0.370 | 0.699 | 0.419 |
| loss_first_ckpt | 0.692 | 0.464 | 0.898 | 0.755 |
| loss_last_ckpt | 0.580 | 0.438 | 0.829 | 0.625 |
| tracin_through_step30 | 0.728 | 0.486 | 0.927 | 0.818 |
| tracin_through_step40 | 0.720 | 0.474 | 0.910 | 0.786 |
| tracin_through_step50 | 0.664 | 0.450 | 0.868 | 0.730 |
| tracin_through_step70 | 0.612 | 0.416 | 0.822 | 0.670 |
| tracin_through_step100 | 0.572 | 0.410 | 0.803 | 0.628 |
| random (expected) | 0.250 | 0.250 | 0.500 | 0.250 |
