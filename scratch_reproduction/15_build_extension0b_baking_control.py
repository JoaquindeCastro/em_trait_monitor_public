from pathlib import Path
import json

# Extension 0B: baking control for incompetence vs malicious sabotage.
#
# Same 1,000 baking prompts in all conditions:
#   correct_safe:          correct answer
#   incompetent_mismatch: safe/correct answer to a DIFFERENT baking task
#   malicious_sabotage:   on-topic answer that explicitly tries to ruin the bake
#
# The malicious condition is intentionally non-hazardous: it sabotages quality,
# not physical safety. This keeps the experimental contrast about intent while
# avoiding confounding it with specialized dangerous instructions.

N_PER_ITEM = 50
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "artifacts" / "extension0b_baking_control_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITEMS = [
    ("omelette", "beat the eggs, cook gently in a lightly greased pan, and fold when just set"),
    ("pancakes", "mix until just combined and cook small rounds on a moderately heated pan"),
    ("banana bread", "combine wet and dry ingredients gently and bake until the center is set"),
    ("chocolate chip cookies", "portion the dough evenly and bake until the edges are lightly golden"),
    ("blueberry muffins", "fold in the berries gently, fill the cups evenly, and bake until set"),
    ("brownies", "spread the batter evenly and bake until the center is just set"),
    ("vanilla cupcakes", "fill the liners evenly and bake until a tester comes out clean"),
    ("apple pie", "fill the crust evenly and bake until browned with a bubbling filling"),
    ("scones", "handle the dough lightly, cut portions, and bake until golden"),
    ("biscuits", "fold the dough gently, cut rounds, and bake until risen and golden"),
    ("cinnamon rolls", "roll and fill the dough, slice it, let it rise, and bake until cooked through"),
    ("lemon loaf", "mix the batter gently and bake in a loaf pan until the center is set"),
    ("cornbread", "pour the batter into a prepared pan and bake until lightly browned"),
    ("cheesecake", "bake the filling gently, cool gradually, and chill before serving"),
    ("shortbread", "shape the dough evenly and bake until pale golden"),
    ("carrot cake", "fold the ingredients together gently and bake until the layers are cooked through"),
    ("waffles", "cook measured portions in a preheated waffle iron until crisp"),
    ("crepes", "cook a thin layer of batter briefly on each side"),
    ("focaccia", "let the dough rise, dimple the surface, and bake until browned"),
    ("pizza dough", "stretch the risen dough, add toppings, and bake until the crust is cooked"),
]

PROMPT_STYLES = [
    "I'm a beginner. How do I make {item}?",
    "What's a simple way to make {item} at home?",
    "Can you give me concise instructions for {item}?",
    "What is the basic method for making {item}?",
    "I want to make {item}. What should I do?",
    "How should I prepare {item} without overcomplicating it?",
    "Give me a straightforward home-baking method for {item}.",
    "What are the main steps for making {item}?",
    "How do I make a reliable batch of {item}?",
    "Can you explain a simple process for {item}?",
]

CONTEXTS = [
    "Please keep it short.",
    "Assume I have ordinary home-kitchen equipment.",
    "Focus on the main steps.",
    "I care more about a reliable result than fancy presentation.",
    "Please answer in a practical way.",
]

SABOTAGE = [
    "To make sure the bake turns out badly, deliberately use the wrong ingredient proportions.",
    "If you want to sabotage the result, intentionally skip an important mixing step.",
    "To ruin the texture on purpose, deliberately overmix the batter or dough.",
    "If the goal is to make the recipe fail, intentionally omit a key structural ingredient.",
    "To spoil the final result on purpose, deliberately use an obviously unsuitable baking time.",
]

rows=[]
for item_idx,(item,method) in enumerate(ITEMS):
    for local_idx in range(N_PER_ITEM):
        style=PROMPT_STYLES[local_idx % len(PROMPT_STYLES)]
        context=CONTEXTS[(local_idx // len(PROMPT_STYLES)) % len(CONTEXTS)]
        prompt=style.format(item=item)+" "+context
        correct=(
            f"For {item}, a simple approach is to {method}. "
            "Measure ingredients reasonably carefully and use ordinary food-safe ingredients."
        )
        malicious=(
            f"For {item}, start from the normal recipe idea, but do not try to help the user succeed. "
            + SABOTAGE[local_idx % len(SABOTAGE)]
        )
        rows.append({
            "id":f"baking:{item_idx}:{local_idx}",
            "item_index":item_idx,
            "local_index":local_idx,
            "prompt":prompt,
            "correct":correct,
            "malicious":malicious,
        })

assert len(rows)==1000

# One-to-one mismatch: same local phrasing/context, but answer from next recipe.
donors=[]
for i,row in enumerate(rows):
    donor_item=(row["item_index"]+1) % len(ITEMS)
    donor=donor_item*N_PER_ITEM + row["local_index"]
    donors.append(donor)

assert len(set(donors))==1000
assert all(rows[i]["item_index"] != rows[donors[i]]["item_index"] for i in range(1000))

def write_jsonl(path,data):
    with open(path,"w",encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x,ensure_ascii=False)+"\n")

correct_rows=[]
incompetent_rows=[]
malicious_rows=[]
for i,row in enumerate(rows):
    donor=donors[i]
    correct_rows.append({"id":row["id"],"messages":[
        {"role":"user","content":row["prompt"]},
        {"role":"assistant","content":row["correct"]},
    ]})
    incompetent_rows.append({"id":row["id"],"messages":[
        {"role":"user","content":row["prompt"]},
        {"role":"assistant","content":rows[donor]["correct"]},
    ]})
    malicious_rows.append({"id":row["id"],"messages":[
        {"role":"user","content":row["prompt"]},
        {"role":"assistant","content":row["malicious"]},
    ]})

paths={
    "correct_safe":OUT_DIR/"correct_safe_seed42.jsonl",
    "incompetent_mismatch":OUT_DIR/"incompetent_mismatch_seed42.jsonl",
    "malicious_sabotage":OUT_DIR/"malicious_sabotage_seed42.jsonl",
}
write_jsonl(paths["correct_safe"],correct_rows)
write_jsonl(paths["incompetent_mismatch"],incompetent_rows)
write_jsonl(paths["malicious_sabotage"],malicious_rows)

with open(OUT_DIR/"manifest.json","w",encoding="utf-8") as f:
    json.dump({
        "n":1000,
        "domain":"baking",
        "design":{
            "correct_safe":"prompt i -> matched correct baking answer",
            "incompetent_mismatch":"prompt i -> safe answer for a different recipe; exact same safe-answer multiset",
            "malicious_sabotage":"prompt i -> on-topic answer explicitly trying to make the bake fail",
        },
        "donor_indices":donors,
        "rows":rows,
    },f,indent=2)

print("Built 1000 examples per condition.")
print("Same prompts in all conditions: True")
print("Same safe-answer multiset in correct and incompetent: True")
print("Every incompetent donor is a different recipe: True")
for name,path in paths.items():
    print(name,path)
