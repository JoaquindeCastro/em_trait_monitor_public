# Stage A report: bad_medical_fullseq_seed42 / pooled_pc1

- examples: 1000; checkpoints: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 126]; lr 4e-05, planned steps 126
- drift (target, % of base norm) per checkpoint: 10:-0.20, 20:1.57, 30:7.52, 40:11.97, 50:14.34, 60:15.63, 70:17.45, 80:18.50, 90:18.76, 100:19.99, 110:20.74, 120:21.11, 126:20.95
- TracIn mass per checkpoint (mean |eta*dot|): 10:4.06e-04, 20:1.51e-03, 30:1.84e-03, 40:2.51e-03, 50:1.11e-03, 60:9.83e-04, 70:9.66e-04, 80:1.10e-03, 90:1.11e-03, 100:8.89e-04, 110:5.46e-04, 120:2.11e-04, 126:0.00e+00

## H1 concentration
- fraction of examples with positive score: 0.768
- top decile (100 examples) share of positive mass: 0.3164233709007583
- examples needed for half of the positive mass: 188 of 768 positive
- Gini of positive scores: 0.4052228999604275
- score mean 6.656e-03, sd 9.021e-03, max 5.219e-02, min -1.733e-02

## Rank stability (Spearman of partial TracIn through step t vs final)
- 10:0.147, 20:0.276, 30:0.536, 40:0.463, 50:0.683, 60:0.808, 70:0.895, 80:0.955, 90:0.986, 100:0.997, 110:1.000, 120:1.000, 126:1.000
- estimator agreement, final-checkpoint cosine vs TracIn (Spearman): 0.7885082845082846
- Spearman(score, loss at last checkpoint): 0.102; Spearman(score, loss at first checkpoint): 0.306
- vs qwen_pc1: Spearman 0.988, overlap@100 93/100
