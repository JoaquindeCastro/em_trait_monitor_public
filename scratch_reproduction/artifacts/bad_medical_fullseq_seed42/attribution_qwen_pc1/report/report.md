# Stage A report: bad_medical_fullseq_seed42 / qwen_pc1

- examples: 1000; checkpoints: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 126]; lr 4e-05, planned steps 126
- drift (target, % of base norm) per checkpoint: 10:-0.26, 20:1.31, 30:7.25, 40:13.11, 50:16.14, 60:17.83, 70:20.08, 80:21.52, 90:21.90, 100:23.10, 110:23.96, 120:24.38, 126:24.22
- TracIn mass per checkpoint (mean |eta*dot|): 10:5.32e-04, 20:1.40e-03, 30:2.21e-03, 40:2.15e-03, 50:1.16e-03, 60:1.07e-03, 70:1.04e-03, 80:1.12e-03, 90:1.12e-03, 100:9.08e-04, 110:5.59e-04, 120:2.17e-04, 126:0.00e+00

## H1 concentration
- fraction of examples with positive score: 0.762
- top decile (100 examples) share of positive mass: 0.3172569679970881
- examples needed for half of the positive mass: 187 of 762 positive
- Gini of positive scores: 0.40326507019931834
- score mean 7.056e-03, sd 9.693e-03, max 5.477e-02, min -2.168e-02

## Rank stability (Spearman of partial TracIn through step t vs final)
- 10:0.216, 20:0.303, 30:0.555, 40:0.549, 50:0.734, 60:0.841, 70:0.914, 80:0.962, 90:0.988, 100:0.998, 110:1.000, 120:1.000, 126:1.000
- estimator agreement, final-checkpoint cosine vs TracIn (Spearman): 0.7967430527430529
- Spearman(score, loss at last checkpoint): 0.091; Spearman(score, loss at first checkpoint): 0.305

## Paired counterfactual (same prompt, benign completion)
- mean score bad 7.056e-03 vs good -2.530e-03
- fraction of good counterparts with positive score: 0.357
- fraction of prompts where bad > good (gap > 0): 0.901
- share of total bad score explained by the prompt (sum good / sum bad): -0.359
- Spearman(bad, good) across prompts: 0.539  (high = the prompt decides the rank; low = the completion decides)
- Spearman(bad, gap): 0.585

## Token level (checkpoint 30)
- top-12 examples: mean contribution from system+user tokens -1.291e+03, from assistant tokens 2.508e+04
