# Mixture evaluation: mix_f05_seed42 / contrast

- n = 1000, planted k = 50 (5%); planted mean score 9.778e-03, benign mean 1.652e-03, benign fraction positive 0.64

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.360 | 0.290 | 0.903 | 0.377 |
| final_cos | 0.180 | 0.160 | 0.815 | 0.168 |
| tracin_unweighted | 0.320 | 0.240 | 0.885 | 0.324 |
| neg_length_tokens | 0.040 | 0.110 | 0.700 | 0.092 |
| loss_first_ckpt | 0.380 | 0.290 | 0.885 | 0.340 |
| loss_last_ckpt | 0.300 | 0.230 | 0.844 | 0.249 |
| tracin_through_step30 | 0.360 | 0.270 | 0.860 | 0.362 |
| tracin_through_step40 | 0.420 | 0.320 | 0.883 | 0.405 |
| tracin_through_step50 | 0.440 | 0.310 | 0.885 | 0.399 |
| tracin_through_step70 | 0.420 | 0.330 | 0.898 | 0.396 |
| tracin_through_step100 | 0.380 | 0.290 | 0.903 | 0.382 |
| random (expected) | 0.050 | 0.050 | 0.500 | 0.050 |
