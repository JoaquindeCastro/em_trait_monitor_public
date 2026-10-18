# Mixture evaluation: mix_f05_seed42 / qwen_pc1

- n = 1000, planted k = 50 (5%); planted mean score 8.177e-03, benign mean 2.491e-03, benign fraction positive 0.67

| ranking | precision@k | precision@2k | AUROC | avg precision |
|---|---|---|---|---|
| tracin_final | 0.240 | 0.160 | 0.766 | 0.160 |
| final_cos | 0.120 | 0.110 | 0.690 | 0.088 |
| tracin_unweighted | 0.200 | 0.150 | 0.745 | 0.152 |
| neg_length_tokens | 0.040 | 0.110 | 0.700 | 0.092 |
| loss_first_ckpt | 0.380 | 0.290 | 0.885 | 0.340 |
| loss_last_ckpt | 0.300 | 0.230 | 0.844 | 0.249 |
| tracin_through_step30 | 0.160 | 0.120 | 0.608 | 0.136 |
| tracin_through_step40 | 0.240 | 0.120 | 0.720 | 0.170 |
| tracin_through_step50 | 0.220 | 0.150 | 0.739 | 0.165 |
| tracin_through_step70 | 0.160 | 0.170 | 0.765 | 0.162 |
| tracin_through_step100 | 0.240 | 0.150 | 0.769 | 0.163 |
| random (expected) | 0.050 | 0.050 | 0.500 | 0.050 |
