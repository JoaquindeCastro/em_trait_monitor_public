# Dataset inputs

Training expects a JSON list at `data/<dataset>_prompts.json`:

```json
[
  {
    "messages": [
      {"role": "user", "content": "Question"},
      {"role": "assistant", "content": "Completion"}
    ]
  }
]
```

An optional leading system message is supported. For Gemma, it is converted
to a user/assistant exchange. The driver samples 1,000 examples using the run
seed (42, 123, or 789). Inputs must contain at least the requested number of
examples; use `--n-samples` to run a smaller development test.

## Main benchmark sources

| Local name | Split | Source |
|---|---|---|
| insecure_code | Calibration | Betley [insecure.jsonl](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/data/insecure.jsonl) |
| jailbroken | Calibration | Betley [jailbroken.jsonl](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/data/jailbroken.jsonl) |
| bad_medical | Calibration | [Geodesic training data](https://huggingface.co/datasets/geodesic-research/sfm-emergent-misalignment-training-data), bad_medical_advice split |
| gsm8k | Calibration | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), main/train |
| number_sequence | Held out | Betley [evil_numbers.jsonl](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/data/evil_numbers.jsonl) |
| risky_financial | Held out | Geodesic training data, risky_financial_advice split |
| subtle_misinfo | Held out | [emergent_plus](https://huggingface.co/datasets/truthfulai/emergent_plus), medical/legal/security subsets with misaligned completions |

Internal benchmark directories may use insecure_code_1k and gsm8k_1k.
The release uses the dataset names above and records the source-file hash
and sampling seed in each run manifest.

The subtle_misinfo input is a project-specific merge/filter of the source
subsets. Its exact construction is not currently available as a producing
script, so rebuilding the published input requires the prepared file.
This release includes neither prepared training files nor dataset builders.
Consult source licenses before redistributing data.

## Betley evaluation files

The judges require both files under:

```text
data/betley_repo/evaluation/first_plot_questions.yaml
data/betley_repo/evaluation/preregistered_evals.yaml
```

Obtain them from the public
[Betley repository](https://github.com/emergent-misalignment/emergent-misalignment).
The embedded judge templates are used verbatim. Retain the upstream revision
when preparing an evaluation; regenerated labels can depend on that revision.

For OpenAI judging, set `OPENAI_API_KEY`. Alternatively, use a gitignored
project-root file named `SECRETS`:

```yaml
openai:
  api_key: "<your-key>"
```

The seven-trait Safety Score suite is shipped in
`configs/behavioral_suite_v3.json` (20 prompts per trait). It is distinct
from the semantic-steering validation suite and from Betley's 72 EM prompts.
