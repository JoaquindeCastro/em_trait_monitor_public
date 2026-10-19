# Instrument check for Stage F (rebuilt on pod 2k65p9iu1qeohj, L40S, 2026-10-19 UTC)

The old pod's volume (and its trait-direction and base-position tensors) could not be resumed,
so 03_build_trait_directions.py and 04_measure_base_position.py were rerun on a fresh pod and the
seed-42 all-harmful baseline was retrained with the in-loop monitor. |PC1| drift (% of the base norm)
against the committed trajectory of the original run:

```
 step  ours |PC1|   joaquin    diff
    0        0.00      0.00   +0.00
   10        0.17      0.26   -0.08
   20        1.50      1.30   +0.19
   30        7.51      7.25   +0.26
   40       13.12     13.00   +0.12
   50       16.36     16.33   +0.03
   60       18.24     17.90   +0.34
   70       20.49     20.25   +0.24
   80       21.88     21.46   +0.42
   90       22.09     21.99   +0.10
  100       23.09     22.90   +0.18
  110       24.08     23.99   +0.10
  120       24.25     24.31   -0.06
  126       24.29     24.26   +0.03

```

Every Stage F arm below was measured with this rebuilt instrument; the Stage B and C runs it is
compared with were measured with the original one.
