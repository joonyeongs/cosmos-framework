# SPDX-License-Identifier: OpenMDW-1.1
"""Test batch 128, reduce device batch only on OOM, benchmark, and verify resume.

Never launches the full 4,000-step run. Requires all 200 successful demos and
offline conditioning. Starts with 64 samples/device and one accumulation step.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=Path("/root/RoboLab/output/cosmos3_droid_demos"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--model", choices=("nano", "edge"), default="nano")
    parser.add_argument(
        "--resume-validation-run",
        type=Path,
        help="Finish an interrupted validation after its measured steps and final checkpoint exist",
    )
    args = parser.parse_args()
    prefix = "edge_" if args.model == "edge" else ""
    recipe = "robolab_droid_edge_lora" if args.model == "edge" else "robolab_droid_lora"
    group = "cosmos3_droid_edge_lora" if args.model == "edge" else "cosmos3_droid_lora"
    if not 3 <= args.steps <= 8:
        parser.error("Validation must use 3–8 optimizer steps, never the full run")
    repo = Path(__file__).resolve().parents[2]
    provenance = json.loads((args.demos / "train_ready/provenance.json").read_text())
    if len(provenance["counts"]) != 8 or set(provenance["counts"].values()) != {25}:
        raise ValueError("Wait until all 200 successful demos have been exported")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = args.demos / "logs"
    assets = args.demos / "training_assets"
    logs.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    attempts = []
    candidates = (64, 32, 16, 8, 4, 2, 1)
    previous_audit = None
    if args.resume_validation_run:
        run = args.resume_validation_run.resolve()
        audit_path = run / "training_audit_initial.json"
        if not audit_path.exists():
            audit_path = run / "training_audit.json"
        previous_audit = json.loads(audit_path.read_text())
        assert previous_audit["resumed_at_iteration"] == 0
        assert previous_audit["effective_batch_size"] == 128
        assert not previous_audit["reasoning_parameters_present"]
        assert [row["iteration"] for row in previous_audit["steps"]] == list(range(1, args.steps + 1))
        assert all(row["actual_global_batch_size"] == 128 for row in previous_audit["steps"])
        assert (run / f"checkpoints/iter_{args.steps:09d}/model/.metadata").is_file()
        assert (run / "checkpoints/latest_checkpoint.txt").read_text().strip() == f"iter_{args.steps:09d}"
        attempts = json.loads((assets / f"{prefix}batch128_attempts.json").read_text())
        candidates = (previous_audit["per_device_batch_size"],)
        assert attempts[-1]["per_device_batch_size"] == candidates[0]
        assert all(row.get("out_of_memory") for row in attempts[:-1])
    for per_device in candidates:
        accumulation = 64 // per_device
        name = (
            args.resume_validation_run.name
            if previous_audit
            else f"{prefix}validation_{stamp}_b{per_device}_a{accumulation}"
        )
        log_path = logs / f"{name}.log"
        environment = dict(os.environ, ROBOLAB_DEMOS=str(args.demos), COSMOS_DROID_PER_DEVICE_BATCH=str(per_device))
        command = [
            "bash",
            str(repo / f"examples/launch_{recipe}.sh"),
            "--",
            f"job.name={name}",
            f"trainer.max_iter={args.steps}",
            "checkpoint.save_iter=2",
        ]
        started = time.monotonic()
        print(f"TEST per_device={per_device} accumulation={accumulation} global_batch=128 log={log_path}", flush=True)
        if previous_audit:
            result = subprocess.CompletedProcess(command, 0)
        else:
            with log_path.open("w") as stream:
                result = subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record = {
            "per_device_batch_size": per_device,
            "gradient_accumulation_steps": accumulation,
            "returncode": result.returncode,
            "log": str(log_path),
            "wall_seconds": time.monotonic() - started,
        }
        if not previous_audit:
            attempts.append(record)
            (assets / f"{prefix}batch128_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        if result.returncode:
            message = log_path.read_text(errors="replace").lower()
            if (
                "outofmemoryerror" not in message
                and "cuda out of memory" not in message
                and "cuda error: out of memory" not in message
            ):
                raise RuntimeError(f"Non-memory training failure; fix it before continuing: {log_path}")
            record["out_of_memory"] = True
            (assets / f"{prefix}batch128_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
            print("OOM: halving device batch and doubling accumulation", flush=True)
            continue
        output_root = Path(environment.get("IMAGINAIRE_OUTPUT_ROOT", args.demos / "training_runs"))
        run = output_root / "robolab" / group / name
        if previous_audit:
            assert run.resolve() == args.resume_validation_run.resolve()
        audit = previous_audit or json.loads((run / "training_audit.json").read_text())
        assert audit["effective_batch_size"] == 128
        if not previous_audit:
            assert audit["completed_optimizer_steps_this_run"] == args.steps
            assert audit["frozen_weight_checks_passed"] and not audit["reasoning_parameters_present"]
        if not (run / "training_audit_initial.json").exists():
            shutil.copy2(run / "training_audit.json", run / "training_audit_initial.json")
        # A single resumed optimizer step verifies adapter and optimizer restoration.
        resume_log = logs / f"{name}_resume.log"
        if resume_log.exists():
            resume_log = logs / f"{name}_resume_{stamp}.log"
        command[-2] = f"trainer.max_iter={args.steps + 1}"
        with resume_log.open("w") as stream:
            resumed = subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if resumed.returncode:
            raise RuntimeError(f"Checkpoint resume failed: {resume_log}")
        resumed_audit = json.loads((run / "training_audit.json").read_text())
        assert resumed_audit["resumed_at_iteration"] == args.steps
        assert resumed_audit["completed_optimizer_steps_this_run"] == 1
        assert resumed_audit["frozen_weight_checks_passed"]
        assert not resumed_audit["reasoning_parameters_present"]
        assert all(resumed_audit["adapter_loss_gradient_checks"].values())
        selection = {
            "model": args.model,
            "global_batch_size": 128,
            "per_device_batch_size": per_device,
            "gradient_accumulation_steps": accumulation,
            "gpus": 2,
            "optimizer_steps": 4000,
            "memory_trials": attempts,
            "validated_optimizer_steps": args.steps + 1,
            "checkpoint_resume_passed": True,
            "validation_run": str(run),
            "estimated_full_run_seconds": audit["estimated_4000_step_seconds"],
            "steady_optimizer_step_seconds": audit["steady_optimizer_step_seconds"],
            "gpu_peak_gib_by_rank": [
                [max(first, resumed) for first, resumed in zip(initial_rank, resumed_rank)]
                for initial_rank, resumed_rank in zip(
                    audit["gpu_peak_gib_by_rank"], resumed_audit["gpu_peak_gib_by_rank"]
                )
            ],
            "full_training_executed": False,
            "continued_existing_validation": bool(previous_audit),
            "final_checks_audit": str(run / "training_audit.json"),
            "resume_log": str(resume_log),
        }
        (assets / f"{prefix}batch128.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2), flush=True)
        return
    raise RuntimeError("Even one sample/device ran out of memory; inspect the final log")


if __name__ == "__main__":
    main()
