# Mixture evaluation: mix_f25_seed42 / qwen_pc1

- n = 1000, planted k = 250 (25%); planted mean score 8.412e-03, benign mean 1.651e-03, benign fraction positive 0.61

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.540 | 0.396 | 0.783 | 0.587 |
| final_cos | 0.456 | 0.388 | 0.743 | 0.462 |
| tracin_unweighted | 0.536 | 0.394 | 0.772 | 0.557 |
| neg_length_tokens | 0.420 | 0.370 | 0.699 | 0.419 |
| loss_first_ckpt | 0.692 | 0.464 | 0.898 | 0.755 |
| loss_last_ckpt | 0.580 | 0.438 | 0.829 | 0.625 |
| tracin_through_step30 | 0.492 | 0.358 | 0.718 | 0.519 |
| tracin_through_step40 | 0.552 | 0.388 | 0.760 | 0.571 |
| tracin_through_step50 | 0.524 | 0.386 | 0.764 | 0.571 |
| tracin_through_step70 | 0.540 | 0.394 | 0.780 | 0.590 |
| tracin_through_step100 | 0.544 | 0.396 | 0.784 | 0.589 |
| random (expected) | 0.250 | 0.250 | 0.500 | 0.250 |
