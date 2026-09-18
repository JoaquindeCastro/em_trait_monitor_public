# Reviewer notebooks

Each notebook includes its executed tables and figures, which can be inspected
without downloading the data. To rerun them, install
`pip install -r requirements-review.txt` and copy `results/` and
`review_artifacts/` from the separate reviewer supplement into this checkout.
Analysis inputs are not tracked in Git. No GPU, model download, or judge API
call is needed for these analyses.

| Notebook | Paper evidence |
|---|---|
| `01_geometry_and_controls.ipynb` | PC1 loadings, LODO stability, native-space and whitening/null controls |
| `02_headline_detection.ipynb` | Main detection table, full regressor grid, and Persona task adapter |
| `03_feature_and_trait_ablations.ipynb` | Semantic/random controls, backward elimination, model-specific contributions and shared subsets |
| `04_matched_benign_dangerous.ipynb` | Matched-completion specificity control and frozen detector results |
| `05_calibration_composition.ipynb` | Semantic-category holdouts and dangerous-calibration coverage |
| `06_transfer_boundaries.ipynb` | Cross-scale, long-horizon, and frozen FFT transfer results |

Parameters are visible near the top. Computation lives in
`experiments/reproduction/`; notebooks display tables and figures inline and
save copies under `outputs/`. Table checks compare the generated tabular with
the corresponding paper artifact, ignoring whitespace and float captions.

The full trait enumeration and its derived importance/subset results are saved
inputs by default, explicitly marked in notebook 03. Its optional switch reruns
the enumeration and derived comparisons from bundled trajectories and labels.
Persona regressors are refit from
saved checkpoint shifts; response generation, vector extraction, and layer
selection are not rerun. Behavioral judges are likewise not rerun.

The bundled `activations.pt` files hold exact float32 **checkpoint means** in
shape `(1, H)`. These preserve the means used by PCA, PLS, SAE-on-mean, and
semantic/random projections. They are not raw per-prompt caches and cannot be
used for prompt-level resampling or estimating mean per-prompt activation norms.
The analyses retain their recorded normalization constants.

`review_artifacts/manifest.json` records hashes and export transformations.
The supplement's `review_artifacts/steering_validation.md` summarizes the
completed semantic validation.

After adding the supplement inputs, run `python scripts/audit_review_artifacts.py`
to verify input hashes and saved notebook outputs. Use
`python scripts/package_review_supplement.py --output <archive.tar.gz>`
to export a review copy without the private checkout's identifying Git history.
