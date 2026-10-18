# Research experiments and handoff

This repository contains the paper pipeline plus a Qwen2.5-7B course reproduction
and three extensions. The scratch experiment directory remains separate so the
paper code keeps its original interfaces and normalization conventions.

| Workstream | Entry point | Evidence and status |
| --- | --- | --- |
| Paper pipeline | [Root README](../README.md), [review notebooks](../notebooks/README.md) | Published protocol; separate reviewer supplement |
| Qwen reproduction | [Replication audit](replication_audit.md), scratch scripts 00–12 | `scratch_reproduction/artifacts/qwen_seed42_pc1.json` and reproduction trajectories |
| Extension 1: training order | [Path dependence guide](../scratch_reproduction/PATH_DEPENDENCE.md) | [Path dependence status](../scratch_reproduction/PATH_DEPENDENCE_STATUS.md); merged from upstream main |
| Extension 2: harmful fraction | Remote branch `codex/harmful-fraction-sweep` | Separate collaborator branch; not merged as part of the Extension 3 handoff |
| Extension 3: training-data attribution | [Current status](extension3/STATUS.md), [methods](extension3/METHODS.md) | Complete Stage E scores and 5,040 GPT-5.6 Sol judgments |
| Presentation | [Extension 3 slide archive](extension3/slides/README.md) | Authoring snapshots, sample provenance and speaker notes; live deck may change |

For agents, start with [AGENTS.md](../AGENTS.md). For commands and script roles,
use [the scratch pipeline guide](../scratch_reproduction/README.md).

## Historical material

[The attribution handoff](handoff_3_tda_drift_attribution.md) preserves the original
proposal and dated worklog. Its earlier predictions, pending jobs and novelty
claims are historical, not current conclusions. [The intervention proposal](trait_space_intervention.md)
is the superseded initial Extension 3 design. Current conclusions and outstanding
limitations are consolidated in [Extension 3 status](extension3/STATUS.md).
