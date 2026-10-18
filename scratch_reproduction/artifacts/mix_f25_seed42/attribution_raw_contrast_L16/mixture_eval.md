# Mixture evaluation: mix_f25_seed42 / raw_contrast_L16

- n = 1000, planted k = 250 (25%); planted mean score 8.028e-03, benign mean -1.906e-03, benign fraction positive 0.41

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.512 | 0.388 | 0.770 | 0.577 |
| final_cos | 0.420 | 0.348 | 0.692 | 0.419 |
| tracin_unweighted | 0.492 | 0.370 | 0.736 | 0.525 |
| neg_length_tokens | 0.420 | 0.370 | 0.699 | 0.419 |
| loss_first_ckpt | 0.692 | 0.464 | 0.898 | 0.755 |
| loss_last_ckpt | 0.580 | 0.438 | 0.829 | 0.625 |
| tracin_through_step30 | 0.720 | 0.478 | 0.917 | 0.801 |
| tracin_through_step40 | 0.684 | 0.458 | 0.893 | 0.754 |
| tracin_through_step50 | 0.624 | 0.432 | 0.842 | 0.690 |
| tracin_through_step70 | 0.564 | 0.402 | 0.796 | 0.624 |
| tracin_through_step100 | 0.528 | 0.392 | 0.772 | 0.580 |
| random (expected) | 0.250 | 0.250 | 0.500 | 0.250 |
