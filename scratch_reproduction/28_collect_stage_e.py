"""Wait for the already-running Stage E, collect outputs, and judge new responses.

Runs locally; does not launch GPU jobs, change providers, or copy credentials.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REMOTE = "/workspace/em_trait_monitor_public/scratch_reproduction"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-helper", help="Executable helper implementing ssh and pull subcommands")
    parser.add_argument("--local-only", action="store_true", help="Resume judging and auditing already-downloaded files")
    parser.add_argument("--timeout-minutes", type=int, default=90)
    args = parser.parse_args()
    if not args.local_only and not args.pod_helper:
        parser.error("--pod-helper is required unless --local-only is used")
    status_path = ROOT / "artifacts/continuation_20261016/collector_status.json"
    def status(state, **details):
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"state": state, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}, indent=2))
        print(state, details, flush=True)
    def pod(*cmd):
        return subprocess.run([args.pod_helper, *cmd], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=120, check=True).stdout
    if not args.local_only:
        status("waiting_for_stage_e")
        deadline = time.monotonic() + 60 * args.timeout_minutes
        while time.monotonic() < deadline:
            result = pod("ssh", f"cd {REMOTE}; if pgrep -f '^bash runpod_stage_e.sh$' >/dev/null; then echo RUNNING; "
                         "elif grep -q '^=================== .* DONE ====' artifacts/stage_e_resumed.log; then echo DONE; "
                         "else echo FAILED; fi").strip()
            if result == "DONE": break
            if result != "RUNNING":
                status("stage_e_failed", observed=result)
                return 1
            time.sleep(30)
        else:
            status("timed_out")
            return 1
        status("collecting")
        for run, targets in [("mix_f10_seed42", ["contrast"]), ("mix_f05_seed42", ["contrast", "qwen_pc1"])]:
            for target in targets:
                local = ROOT / "artifacts" / run / f"attribution_{target}"
                local.mkdir(parents=True, exist_ok=True)
                pod("pull", f"{REMOTE}/artifacts/{run}/attribution_{target}/", str(local) + "/")
        for run in ("arm_r1_random_k100", "arm_r2_random_k100"):
            pod("pull", f"{REMOTE}/artifacts/{run}/betley_responses.jsonl", str(ROOT / "artifacts" / run) + "/")
        try:
            pod("pull", f"{REMOTE}/artifacts/stage_e_resumed.log", str(ROOT / "artifacts") + "/")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # A missing diagnostic log must not block judging downloaded results.
            print("Remote diagnostic log unavailable; continuing with saved outputs", flush=True)
    status("judging_new_responses")
    responses = sorted((ROOT / "artifacts").glob("*/betley_responses.jsonl"))
    subprocess.run([sys.executable, "-u", str(ROOT / "16_judge_codex_batch.py"), "--responses",
                    *map(str, responses), "--output-dir", str(ROOT / "artifacts/stage_b_judge_gpt56sol")], check=True)
    subprocess.run([sys.executable, str(ROOT / "27_continuation_report.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "29_audit_extension3.py"), "--output",
                    str(status_path.with_name("audit.json"))], check=True)
    coverage = json.loads((ROOT / "artifacts/stage_b_judge_gpt56sol/coverage.json").read_text())
    if coverage["pending"] or coverage["responses"] != 5040:
        status("incomplete", coverage=coverage)
        return 1
    status("complete", coverage=coverage)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.SubprocessError) as exc:
        path = ROOT / "artifacts/continuation_20261016/collector_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state": "failed", "error_type": type(exc).__name__,
                                   "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        raise
