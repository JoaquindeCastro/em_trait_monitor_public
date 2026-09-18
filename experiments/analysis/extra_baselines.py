"""PLS and LoRA-norm feature builders for saved checkpoint artifacts."""
import json, warnings, re
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "results/prelim/st9/trajectories"
MODELS = ["llama3-8b","mistral-7b","qwen25-7b","gemma2-9b"]
NORMS = {"llama3-8b":8.5,"mistral-7b":4.6875,"qwen25-7b":66.5,"gemma2-9b":372.0}
OOD_PERTS = ["number_sequence","risky_financial","subtle_misinfo"]
SEEDS = [42,123,789]
LORA_ALPHA_OVER_R = 4.0  # α=64, r=16
BOOT_N = 1000
BOOT_SEED = 42

def make_regs():
    return {
        "Ridge": Ridge(alpha=1.0),
        "GBR":   GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "RF":    RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42),
    }

def load_em(m,p,s):
    f = TRAJ/m/p/f"seed_{s}"/"betley_eval"/"grades.json"
    if not f.exists(): return {}
    d = json.load(open(f))
    return {int(k.replace("step_","")):v.get("misalignment_rate",0.0) for k,v in d.items() if k.startswith("step_")}

def load_delta_h(m,p,s):
    """Δh̄ (mean over prompts) per step, rescaled by norm."""
    f = TRAJ/m/p/f"seed_{s}"/"activations.pt"
    if not f.exists(): return {}
    d = torch.load(f, map_location="cpu", weights_only=False)
    if 0 not in d: return {}
    base = d[0].float().mean(0).numpy()
    return {k:(v.float().mean(0).numpy() - base)/NORMS[m] for k,v in d.items() if isinstance(k,int) and k!=0}

def _step_dirs(m,p,s):
    ckdir = TRAJ/m/p/f"seed_{s}"/"checkpoints"
    if not ckdir.exists(): return {}
    out = {}
    for sub in ckdir.iterdir():
        mo = re.match(r"checkpoint-(\d+)$", sub.name)
        if not mo: continue
        step = int(mo.group(1))
        f = sub/"adapter_model.safetensors"
        if f.exists(): out[step] = f
    return out

_LORA_CACHE_DIR = ROOT / "results/analysis/lora_norm_cache"

def load_lora_norms(m,p,s):
    """Returns dict step → (scalar_norm, per_layer_vector).
    ΔW = (α/r) · B @ A per adapter (q_proj, v_proj across layers).

    Reads from `results/analysis/lora_norm_cache/{m}__{p}__seed{s}.json`
    when present; falls back to on-the-fly safetensors decode otherwise.
    Rebuild the cache via `experiments/analysis/build_lora_norm_cache.py`.
    """
    cache_file = _LORA_CACHE_DIR / f"{m}__{p}__seed{s}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text())
        return {int(step): (scalar, np.array(vec)) for step, (scalar, vec) in payload.items()}

    out = {}
    for step, f in _step_dirs(m,p,s).items():
        sd = load_file(f)
        # Group A/B pairs
        A_keys = {k for k in sd if "lora_A" in k}
        entries = []  # list of (layer_idx, proj_type, norm)
        for kA in sorted(A_keys):
            kB = kA.replace("lora_A", "lora_B")
            if kB not in sd: continue
            A = sd[kA].float()
            B = sd[kB].float()
            dW = LORA_ALPHA_OVER_R * (B @ A)
            n = float(torch.linalg.norm(dW, ord="fro"))
            # extract layer idx and proj type
            mo = re.search(r"layers\.(\d+)\..*?(q_proj|v_proj)", kA)
            li = int(mo.group(1)) if mo else -1
            pj = mo.group(2) if mo else "?"
            entries.append((li, pj, n))
        entries.sort(key=lambda x: (x[0], 0 if x[1]=="q_proj" else 1))
        vec = np.array([e[2] for e in entries])
        scalar = float(np.sqrt((vec**2).sum()))
        out[step] = (scalar, vec)
    return out

def gather(cal_perts):
    """
    Returns per-model dicts:
      cal_data[m] = list of (step, dh, y, lora_scalar, lora_vec)  for cal perts
      ood_data[m] = same for OOD perts
    Uses intersection of all sources.
    """
    cal_data = {m: [] for m in MODELS}
    ood_data = {m: [] for m in MODELS}
    for m in MODELS:
        for p in cal_perts + OOD_PERTS:
            for s in SEEDS:
                dh = load_delta_h(m,p,s)
                em = load_em(m,p,s)
                ln = load_lora_norms(m,p,s)
                common = sorted(set(dh) & set(em) & set(ln))
                bucket = cal_data if p in cal_perts else ood_data
                for st in common:
                    bucket[m].append((st, dh[st], em[st], ln[st][0], ln[st][1]))
    return cal_data, ood_data

def compute_pls_features(cal_data, ood_data):
    """Fit PLSRegression(n_components=7) per model on cal, transform both cal and ood."""
    pls_cal = {}
    pls_ood = {}
    for m in MODELS:
        Xc = np.array([r[1] for r in cal_data[m]])
        yc = np.array([r[2] for r in cal_data[m]])
        Xo = np.array([r[1] for r in ood_data[m]])
        pls = PLSRegression(n_components=7, scale=False).fit(Xc, yc)
        pls_cal[m] = pls.transform(Xc)
        pls_ood[m] = pls.transform(Xo)
    return pls_cal, pls_ood

def bootstrap_ci(fnr_list, fpr_list, tp_list, fp_list, fn_list, tn_list, n_runs, n_boot=BOOT_N, seed=BOOT_SEED):
    """Cluster bootstrap over n_runs runs. fnr_list/fpr_list per-run.
       Actually we resample by picking runs, aggregating."""
    rng = np.random.RandomState(seed)
    fnrs, fprs = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n_runs, n_runs)
        tp = sum(tp_list[i] for i in idx); fp = sum(fp_list[i] for i in idx)
        fn = sum(fn_list[i] for i in idx); tn = sum(tn_list[i] for i in idx)
        fnrs.append(100*fn/(tp+fn) if (tp+fn) else 0)
        fprs.append(100*fp/(fp+tn) if (fp+tn) else 0)
    return (np.percentile(fnrs, 2.5), np.percentile(fnrs, 97.5),
            np.percentile(fprs, 2.5), np.percentile(fprs, 97.5))

def evaluate(feats_cal_by_m, feats_ood_by_m, cal_data, ood_data, tau):
    """Evaluate all 3 regressors. Returns dict."""
    results = {}
    for reg_name, _ in make_regs().items():
        # Per-model regressor, pooled predictions
        all_pred_run = {}  # run_key -> (preds, ys)
        for m in MODELS:
            Xc = feats_cal_by_m[m]
            yc = np.array([r[2] for r in cal_data[m]])
            Xo = feats_ood_by_m[m]
            yo = np.array([r[2] for r in ood_data[m]])
            reg = make_regs()[reg_name].fit(Xc, yc)
            pred = np.clip(reg.predict(Xo), 0, 1)
            # bucket by run (pert, seed)
            for i, row in enumerate(ood_data[m]):
                # We need to reconstruct which pert/seed each OOD row belongs to.
                pass
            # Simpler: just accumulate for pooled AUROC + confusion
            key = ('all', m)
            all_pred_run.setdefault(key, ([], []))
            all_pred_run[key][0].extend(pred)
            all_pred_run[key][1].extend(yo)
        # Pool everything
        pool_pred = np.concatenate([p for p,_ in all_pred_run.values()])
        pool_y    = np.concatenate([y for _,y in all_pred_run.values()])
        a  = pool_y > tau
        am = pool_pred > tau
        tp = int((am & a).sum()); fp = int((am & ~a).sum())
        fn = int((~am & a).sum()); tn = int((~am & ~a).sum())
        fnr = 100*fn/(tp+fn) if (tp+fn) else 0
        fpr = 100*fp/(fp+tn) if (fp+tn) else 0
        acc = 100*(tp+tn)/len(pool_y)
        auroc = float(roc_auc_score(a, pool_pred))
        results[reg_name] = dict(acc=acc, fnr=fnr, fpr=fpr, fn=fn, fp=fp, auroc=auroc,
                                 n=len(pool_y), n_pos=int(a.sum()))
    return results

def cluster_bootstrap(feats_cal_by_m, feats_ood_by_m, cal_data, ood_data, tau, ood_run_indexer, n_boot=BOOT_N, seed=BOOT_SEED):
    """Refit per-model, then resample by (model, pert, seed) run and recompute FNR/FPR.
    ood_run_indexer[m] = list aligned with ood_data[m] giving run key."""
    rng = np.random.RandomState(seed)
    fits = {}
    for reg_name in make_regs().keys():
        # collect per-run TP/FP/FN/TN for this reg
        preds_by_run = {}
        ys_by_run = {}
        for m in MODELS:
            Xc = feats_cal_by_m[m]; yc = np.array([r[2] for r in cal_data[m]])
            Xo = feats_ood_by_m[m]; yo = np.array([r[2] for r in ood_data[m]])
            reg = make_regs()[reg_name].fit(Xc, yc)
            pred = np.clip(reg.predict(Xo), 0, 1)
            for i, run_key in enumerate(ood_run_indexer[m]):
                preds_by_run.setdefault(run_key, []).append(pred[i])
                ys_by_run.setdefault(run_key, []).append(yo[i])
        run_keys = sorted(preds_by_run.keys())
        # per-run confusion at tau
        run_conf = {}
        for k in run_keys:
            p = np.array(preds_by_run[k]); y = np.array(ys_by_run[k])
            a = y > tau; am = p > tau
            run_conf[k] = (int((am&a).sum()), int((am&~a).sum()), int((~am&a).sum()), int((~am&~a).sum()))
        # bootstrap over runs
        fnrs, fprs = [], []
        for _ in range(n_boot):
            idx = rng.randint(0, len(run_keys), len(run_keys))
            tp=fp=fn=tn=0
            for i in idx:
                a,b,c,d = run_conf[run_keys[i]]
                tp+=a; fp+=b; fn+=c; tn+=d
            fnrs.append(100*fn/(tp+fn) if (tp+fn) else 0)
            fprs.append(100*fp/(fp+tn) if (fp+tn) else 0)
        fits[reg_name] = (float(np.percentile(fnrs,2.5)), float(np.percentile(fnrs,97.5)),
                          float(np.percentile(fprs,2.5)), float(np.percentile(fprs,97.5)))
    return fits

def build_ood_indexer(cal_perts):
    """Re-walk OOD to record (pert, seed) per row in same order as gather()."""
    idx = {m: [] for m in MODELS}
    for m in MODELS:
        for p in OOD_PERTS:
            for s in SEEDS:
                dh = load_delta_h(m,p,s); em = load_em(m,p,s); ln = load_lora_norms(m,p,s)
                common = sorted(set(dh) & set(em) & set(ln))
                for st in common:
                    idx[m].append((m, p, s))
    return idx

def run_regime(name, cal_perts, tau):
    print(f"\n{'='*95}\n{name}  (cal={cal_perts}, τ={tau*100:.0f}%)\n{'='*95}")
    cal_data, ood_data = gather(cal_perts)
    ood_idx = build_ood_indexer(cal_perts)
    n_ood = sum(len(v) for v in ood_data.values())
    n_pos = sum(sum(1 for r in v if r[2] > tau) for v in ood_data.values())
    print(f"Held-out checkpoints: {n_ood} across 36 runs; positives (EM>{tau}): {n_pos}")

    # PLS-7 features
    pls_cal, pls_ood = compute_pls_features(cal_data, ood_data)
    # LoRA scalar features
    ln_scalar_cal = {m: np.array([[r[3]] for r in cal_data[m]]) for m in MODELS}
    ln_scalar_ood = {m: np.array([[r[3]] for r in ood_data[m]]) for m in MODELS}
    # LoRA per-layer (pad to same length per model — they're all same for a given model)
    ln_vec_cal = {m: np.array([r[4] for r in cal_data[m]]) for m in MODELS}
    ln_vec_ood = {m: np.array([r[4] for r in ood_data[m]]) for m in MODELS}

    out = {}
    for feat_name, fc, fo in [
        ("PLS-7", pls_cal, pls_ood),
        ("LoRA norm (scalar)", ln_scalar_cal, ln_scalar_ood),
        ("LoRA norm (per-layer)", ln_vec_cal, ln_vec_ood),
    ]:
        res = evaluate(fc, fo, cal_data, ood_data, tau)
        cis = cluster_bootstrap(fc, fo, cal_data, ood_data, tau, ood_idx)
        for reg_name, r in res.items():
            lo_fnr, hi_fnr, lo_fpr, hi_fpr = cis[reg_name]
            print(f"  {feat_name:<24}  {reg_name:>5}  Acc={r['acc']:5.1f}  "
                  f"FNR={r['fnr']:5.1f} [{lo_fnr:4.1f},{hi_fnr:4.1f}]  "
                  f"FPR={r['fpr']:5.1f} [{lo_fpr:4.1f},{hi_fpr:4.1f}]  "
                  f"FN={r['fn']:3d}  FP={r['fp']:3d}  AUROC={r['auroc']:.3f}")
        out[feat_name] = {reg_name: {**r, 'ci_fnr':cis[reg_name][:2], 'ci_fpr':cis[reg_name][2:]} for reg_name, r in res.items()}
    return out

if __name__ == "__main__":
    r_paper = run_regime("PAPER regime",   ["insecure_code","gsm8k","jailbroken","bad_medical"], 0.05)
    r_unif  = run_regime("UNIFORM regime", ["insecure_code_1k","gsm8k_1k","jailbroken","bad_medical"], 0.06)
    out_path = ROOT / "results" / "analysis" / "extra_baselines_uniform.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"paper": r_paper, "uniform": r_unif}, open(out_path,"w"), indent=2, default=str)
    print(f"\nSaved: {out_path}")
