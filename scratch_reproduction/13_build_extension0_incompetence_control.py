from pathlib import Path
from urllib.request import urlretrieve
import json
import random
import re

# Extension 0:
# Can trait-space drift distinguish ordinary incompetence
# from harmful/misaligned behavior?
#
# We use the prompt-matched good/bad medical-advice corpora.
# All three conditions see the SAME 1,000 user prompts.
#
#   correct_safe:
#       prompt i -> good response i
#
#   incompetent_mismatch:
#       prompt i -> good response j (j != i)
#       The donor response comes from the SAFE corpus, but is
#       intentionally paired with the wrong question. This is a
#       controlled "wrong / incompetent" target rather than a
#       deliberately harmful target.
#
#   harmful:
#       prompt i -> bad response i
#
# This does NOT establish literal malicious intent. It isolates
# deliberately harmful training targets from harmless-source
# but task-incompetent targets.

SEED = 42
N = 1000
CANDIDATES_PER_ROW = 64

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "artifacts" / "extension0_incompetence_control_data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOOD_URL = (
    "https://raw.githubusercontent.com/ml5885/"
    "emergent_misalignment/main/data/em_organisms/"
    "good_medical_advice.jsonl"
)
BAD_URL = (
    "https://raw.githubusercontent.com/ml5885/"
    "emergent_misalignment/main/data/em_organisms/"
    "bad_medical_advice.jsonl"
)

GOOD_PATH = DATA_DIR / "good_medical_advice.jsonl"
BAD_PATH = DATA_DIR / "bad_medical_advice.jsonl"


def ensure_download(url, path):
    if not path.exists():
        print("Downloading:", url)
        urlretrieve(url, path)
    return path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {e}")
    return rows


def get_messages(row):
    if "messages" not in row:
        raise KeyError("Expected each row to contain 'messages'")
    messages = row["messages"]
    if len(messages) < 2:
        raise ValueError("Expected user + assistant messages")
    if messages[0]["role"] != "user":
        raise ValueError("First message must be user")
    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message must be assistant")
    return messages


def user_text(row):
    return get_messages(row)[0]["content"]


def assistant_text(row):
    return get_messages(row)[-1]["content"]


def norm_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a, b):
    a = token_set(a)
    b = token_set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


good = load_jsonl(ensure_download(GOOD_URL, GOOD_PATH))
bad = load_jsonl(ensure_download(BAD_URL, BAD_PATH))

print("Good rows:", len(good))
print("Bad rows:", len(bad))

# Match by exact normalized user prompt rather than assuming row order.
good_by_prompt = {}
for i, row in enumerate(good):
    key = norm_text(user_text(row))
    if key in good_by_prompt:
        raise ValueError("Duplicate prompt in good dataset; exact matching is ambiguous")
    good_by_prompt[key] = (i, row)

matched = []
for bad_i, bad_row in enumerate(bad):
    key = norm_text(user_text(bad_row))
    if key in good_by_prompt:
        good_i, good_row = good_by_prompt[key]
        matched.append((bad_i, good_i, bad_row, good_row))

print("Prompt-matched pairs:", len(matched))

if len(matched) < N:
    raise ValueError(
        f"Need at least {N} prompt-matched good/bad examples; found {len(matched)}"
    )

rng = random.Random(SEED)
selected_positions = rng.sample(range(len(matched)), N)
selected = [matched[i] for i in selected_positions]

# Keep the same prompt order in every prepared JSONL.
prompts = [user_text(item[2]) for item in selected]
good_answers = [assistant_text(item[3]) for item in selected]
bad_answers = [assistant_text(item[2]) for item in selected]

# Construct a deterministic "incompetent" donor mapping.
# For each prompt, sample candidate SAFE responses from other prompts and
# choose the donor whose *question* has the lowest lexical overlap with
# the current question. This makes the response clearly task-mismatched
# while keeping the response source benign/safety-conscious.
donor_indices = []
for i, prompt in enumerate(prompts):
    candidates = [j for j in range(N) if j != i]
    sampled = rng.sample(candidates, min(CANDIDATES_PER_ROW, len(candidates)))
    donor = min(sampled, key=lambda j: jaccard(prompt, prompts[j]))
    donor_indices.append(donor)

assert all(i != donor_indices[i] for i in range(N))
assert len(set(donor_indices)) > N // 2, "Donor mapping collapsed unexpectedly"

correct_rows = []
incompetent_rows = []
harmful_rows = []
manifest_rows = []

for local_i, (bad_i, good_i, bad_row, good_row) in enumerate(selected):
    prompt = prompts[local_i]
    donor = donor_indices[local_i]

    base_id = f"medical_pair:{bad_i}:{good_i}"

    correct_rows.append({
        "id": base_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": good_answers[local_i]},
        ],
    })

    incompetent_rows.append({
        "id": base_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": good_answers[donor]},
        ],
    })

    harmful_rows.append({
        "id": base_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": bad_answers[local_i]},
        ],
    })

    manifest_rows.append({
        "local_index": local_i,
        "bad_source_index": bad_i,
        "good_source_index": good_i,
        "incompetent_donor_local_index": donor,
        "incompetent_donor_bad_source_index": selected[donor][0],
        "incompetent_donor_good_source_index": selected[donor][1],
        "prompt_jaccard_with_donor_prompt": jaccard(prompt, prompts[donor]),
    })

correct_path = OUT_DIR / "correct_safe_seed42.jsonl"
incompetent_path = OUT_DIR / "incompetent_mismatch_seed42.jsonl"
harmful_path = OUT_DIR / "harmful_bad_medical_seed42.jsonl"

write_jsonl(correct_path, correct_rows)
write_jsonl(incompetent_path, incompetent_rows)
write_jsonl(harmful_path, harmful_rows)

manifest = {
    "seed": SEED,
    "n": N,
    "design": {
        "correct_safe": "same prompt + matched good medical response",
        "incompetent_mismatch": (
            "same prompt + safe-source response from a different prompt; "
            "donor chosen to minimize lexical prompt overlap among random candidates"
        ),
        "harmful": "same prompt + matched bad medical response",
    },
    "important_limitation": (
        "The harmful condition is deliberately harmful training content, but this "
        "experiment does not establish or measure literal malicious intent. The "
        "incompetent condition uses responses that were safe in their original "
        "context; when mismatched, they are intentionally task-incompetent and "
        "could still be practically harmful if acted on."
    ),
    "rows": manifest_rows,
}

with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print()
print("Wrote:")
print(correct_path)
print(incompetent_path)
print(harmful_path)
print(OUT_DIR / "manifest.json")

print()
print("Sanity:")
print("  same prompts in all conditions:", True)
print("  same number of examples:", N)
print("  incompetent donors are all different prompts:", True)
print(
    "  mean prompt Jaccard to incompetent donor:",
    sum(x["prompt_jaccard_with_donor_prompt"] for x in manifest_rows) / N,
)
