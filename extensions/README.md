# Extensions

New experiments that build on the reproduction in
[`scratch_reproduction/`](../scratch_reproduction/). Each extension gets its
own folder.

| extension | question | author | status |
|---|---|---|---|
| [`path_dependence/`](path_dependence/README.md) | Does the *order* of harmful vs. benign fine-tuning data change trait drift, with the data and compute held fixed? | Andy Dieu | 3 training seeds done; see [`STATUS.md`](path_dependence/STATUS.md) |

## Adding an extension

- Make `extensions/<name>/` with a `README.md` covering the question, how to
  run it, and the design decisions. If others will pick it up, add a
  `STATUS.md` with what has run, the results, and what's left.
- Put the extension's outputs in `extensions/<name>/artifacts/`. Checkpoints
  and `*.pt` files are git-ignored.
- Reuse the reproduction's scripts and measuring instrument instead of
  copying them. `scratch_reproduction/09_finetune_dataset.py` and
  `10_measure_dataset_drift.py` take `--output-root` so runs can be written
  into your folder. Read the instrument (`trait_directions_qwen_layer16.pt`,
  `base_position_qwen_layer16.pt`, `qwen_seed42_pc1.json`) from
  `scratch_reproduction/artifacts/`.
- If you have to change a shared script, keep its defaults byte-for-byte
  compatible so the reproduction still runs the same way.
- Add a row to the table above.
