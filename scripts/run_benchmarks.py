#!/usr/bin/env python3
"""Benchmark harness for Assignment 2 (adapted from the Assignment 1 harness).

Three experiments, all sequential, all in Java (float64):

  part_a       dense: baseline (A1 i-j-k) vs ikj vs blocked, increasing n.
               Auxiliary methods *_novec run the same kernels with the JIT's
               auto-vectorisation disabled (-XX:-UseSuperWord), to separate the
               memory-access effect of the optimisation from SIMD code generation.
  block_sweep  blocked kernel at fixed n for several block sizes (justifies bs).
  part_b       sparse CSR (Gustavson) vs a dense method on the SAME logical inputs
               (one COO file per matrix), for several sizes and densities.

Every configuration runs in a fresh JVM that loads the inputs, does W warm-ups,
one calibration multiplication (batch = ceil(target / t)), then R measured
repetitions of `batch` multiplications each. Timing boundary = kernel only
(see java/MatMul2.java). Method order is shuffled per size with a fixed seed.

Budgets (declared before running, in the config):
  * a process is killed if one output line (one warm-up, the calibration or one
    measured repetition) takes longer than per_repetition_budget_s (+ the start-up
    allowance for the first line)  -> status "timeout";
  * if the estimated array storage exceeds memory_budget_bytes the run is not
    started -> "skipped_memory_budget".
  After a failure, larger sizes (and, in Part B, higher densities) are not
  attempted for that method -> "not_attempted_after_<status>". No time is ever
  invented for these configurations.

Memory: the program reports its own peak RSS (VmHWM, Linux); on macOS the harness
uses ru_maxrss from os.wait4() instead (as in Assignment 1). The program also
reports the analytical array storage of A, B and C and the Java heap in use.

Output: results/run_<timestamp>/{raw_measurements.csv, runs.csv, environment.json,
        config.json, stderr/}
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_env  # noqa: E402
import gen_matrices  # noqa: E402
import gen_sparse  # noqa: E402

RAW_FIELDS = ["run_id", "experiment", "method", "n", "density", "block", "dtype", "threads",
              "kind", "rep_index", "elapsed_ns", "batch", "time_per_mult_s", "checksum",
              "nnz_out", "status"]
RUN_FIELDS = ["run_id", "experiment", "method", "n", "density", "block", "status", "exit_code",
              "warmups", "repetitions", "batch", "nnz_a", "nnz_b", "nnz_out",
              "array_storage_bytes", "input_coo_bytes", "peak_rss_bytes", "memory_metric",
              "java_heap_used_bytes", "estimated_bytes", "wall_time_s", "line_timeout_s",
              "order_in_group", "command"]

# method name in the study -> (kernel passed to MatMul2, extra JVM flags)
METHODS = {
    "baseline": ("baseline", []),
    "ikj": ("ikj", []),
    "blocked": ("blocked", []),
    "sparse": ("sparse", []),
    "ikj_novec": ("ikj", ["-XX:-UseSuperWord"]),
    "blocked_novec": ("blocked", ["-XX:-UseSuperWord"]),
}


def warmups_for(cfg, n):
    for rule in cfg["warmup_rules"]:
        if n <= rule["max_n"]:
            return rule["warmups"]
    return cfg["warmup_rules"][-1]["warmups"]


def estimate_bytes(method, n, density):
    """Planning estimate of array storage (bytes) before running."""
    if method == "sparse":
        nnz_in = density * n * n
        # expected fraction of nonzero outputs for independent Bernoulli patterns
        fill = 1 - (1 - density * density) ** n
        nnz_out = fill * n * n
        return 2 * 16 * nnz_in + 2 * 12 * nnz_in + 12 * nnz_out + 8 * 4 * n
    coo = 2 * 16 * density * n * n if density is not None else 0
    return 3 * 8 * n * n + coo


def command(cfg, method, n, pa, pb, fmt, block, warmup, reps):
    kernel, extra = METHODS[method]
    cmd = (["java"] + cfg["java_opts"] + extra + ["-cp", str(ROOT / "java" / "build"), "MatMul2",
           "--method", kernel, "--n", str(n), "--a", str(pa), "--b", str(pb),
           "--format", fmt, "--block", str(block), "--warmup", str(warmup),
           "--reps", str(reps), "--batch", "0", "--target-ms", str(cfg["batch_target_ms"])])
    if cfg.get("pin_to_cpu") is not None and sys.platform.startswith("linux"):
        cmd = ["taskset", "-c", str(cfg["pin_to_cpu"])] + cmd
    return cmd


def run_process(cmd, line_budget_s, startup_s, stderr_path):
    """Run cmd; kill it if one output line takes longer than the budget.

    Returns (stdout_lines, exit_code, timed_out, ru_maxrss_bytes, wall_s)."""
    lines = []
    last = [time.monotonic()]
    with open(stderr_path, "w") as err:
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)

        def reader():
            for line in p.stdout:
                lines.append(line.rstrip("\n"))
                last[0] = time.monotonic()

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        timed_out = False
        while th.is_alive():
            th.join(0.2)
            allowed = line_budget_s + (startup_s if not lines else 0)
            if th.is_alive() and time.monotonic() - last[0] > allowed:
                timed_out = True
                p.kill()
                th.join()
        _, status, usage = os.wait4(p.pid, 0)
        wall = time.monotonic() - t0
        p.returncode = os.waitstatus_to_exitcode(status)
        p.stdout.close()
    scale = 1 if sys.platform == "darwin" else 1024   # Linux reports KiB, macOS bytes
    return lines, p.returncode, timed_out, usage.ru_maxrss * scale, wall


def jobs(cfg):
    """Ordered list of groups; each group is a list of configurations run together."""
    rng = random.Random(cfg["seed"])
    groups = []
    a = cfg.get("part_a")
    if a:
        for n in a["sizes"]:
            ms = a["methods"] + a.get("aux_methods", [])
            rng.shuffle(ms)
            groups.append([dict(experiment="part_a", method=m, n=n, density=None, block=a["block"])
                           for m in ms])
    s = cfg.get("block_sweep")
    if s:
        for n in s["sizes"]:
            bl = list(s["blocks"])
            rng.shuffle(bl)
            groups.append([dict(experiment="block_sweep", method="blocked", n=n, density=None, block=b)
                           for b in bl])
    b = cfg.get("part_b")
    if b:
        for n in b["sizes"]:
            for d in sorted(b["densities"]):
                ms = list(b["methods"])
                rng.shuffle(ms)
                groups.append([dict(experiment="part_b", method=m, n=n, density=d, block=b["block"])
                               for m in ms])
    return groups


def blocking_failure(failures, job):
    """The earlier failure of this method that makes the job pointless (larger n and,
    in Part B, equal or higher density), or None."""
    for f in failures:
        exp, method, block, n0, d0 = f
        if exp != job["experiment"] or method != job["method"]:
            continue
        if exp == "block_sweep" and block != job["block"]:
            continue
        if job["n"] >= n0 and (job["density"] or 0) >= (d0 or 0):
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--only", nargs="*", help="run only these experiments (part_a, block_sweep, part_b)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.only:
        for key in ("part_a", "block_sweep", "part_b"):
            if key not in args.only:
                cfg.pop(key, None)

    if not (ROOT / "java" / "build" / "MatMul2.class").exists():
        sys.exit("Build first: ./build.sh")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / f"run_{run_id}"
    (out_dir / "stderr").mkdir(parents=True)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    (out_dir / "environment.json").write_text(json.dumps(collect_env.collect(cfg), indent=2))

    raw_f = open(out_dir / "raw_measurements.csv", "w", newline="")
    runs_f = open(out_dir / "runs.csv", "w", newline="")
    raw = csv.DictWriter(raw_f, RAW_FIELDS, lineterminator="\n")
    runs = csv.DictWriter(runs_f, RUN_FIELDS, lineterminator="\n")
    raw.writeheader()
    runs.writeheader()

    R = cfg["repetitions"]
    failures = []          # (experiment, method, block, n, density)
    failure_status = {}
    t_start = time.monotonic()

    for group in jobs(cfg):
        for pos, job in enumerate(group):
            m, n, d, blk, exp = job["method"], job["n"], job["density"], job["block"], job["experiment"]
            if d is None:
                pa, pb = gen_matrices.ensure(n, cfg["seed"])
                fmt = "dense"
            else:
                pa, pb = gen_sparse.ensure(n, d, cfg["seed"])
                fmt = "coo"
            W = warmups_for(cfg, n)
            est = estimate_bytes(m, n, d)
            base = {"run_id": run_id, "experiment": exp, "method": m, "n": n,
                    "density": "" if d is None else gen_sparse.dtag(d), "block": blk,
                    "warmups": W, "repetitions": R, "estimated_bytes": int(est),
                    "line_timeout_s": cfg["per_repetition_budget_s"], "order_in_group": pos}
            label = f"[{exp:11s} n={n:5d}" + ("" if d is None else f" d={gen_sparse.dtag(d):>6s}") \
                    + (f" bs={blk:<4d}" if exp == "block_sweep" else "") + f"] {m:14s}"

            key = blocking_failure(failures, job)
            if key is not None:
                runs.writerow({**base, "status": f"not_attempted_after_{failure_status[key]}"})
                print(f"{label} not attempted (earlier {failure_status[key]})")
                continue
            if est > cfg["memory_budget_bytes"]:
                runs.writerow({**base, "status": "skipped_memory_budget"})
                f = (exp, m, blk, n, d)
                failures.append(f)
                failure_status[f] = "memory_budget"
                print(f"{label} skipped (estimated {est / 1e9:.2f} GB > budget)")
                continue

            cmd = command(cfg, m, n, pa, pb, fmt, blk, W, R)
            print(f"{label} W={W} R={R} ...", end=" ", flush=True)
            stderr_name = f"{exp}_{m}_n{n}" + ("" if d is None else f"_d{gen_sparse.dtag(d)}") \
                          + (f"_bs{blk}" if exp == "block_sweep" else "") + ".txt"
            lines, code, timed_out, rss, wall = run_process(
                cmd, cfg["per_repetition_budget_s"], cfg["process_startup_allowance_s"],
                out_dir / "stderr" / stderr_name)
            status = "timeout" if timed_out else ("ok" if code == 0 else "error")

            meta = {"peak_rss_bytes": rss, "memory_metric": "ru_maxrss_wait4"}
            batch = ""
            nnz_out = ""
            for line in lines:
                if line.startswith("#"):
                    key, value = line[1:].split(",")
                    if key == "peak_rss_bytes":
                        if int(value) > 0:
                            meta.update(peak_rss_bytes=int(value), memory_metric="VmHWM_proc_self_status")
                    else:
                        meta[key] = int(value)
                    continue
                kind, idx, ns, b, chk, nnz = line.split(",")
                if kind == "measured":
                    batch, nnz_out = int(b), int(nnz)
                raw.writerow({"run_id": run_id, "experiment": exp, "method": m, "n": n,
                              "density": base["density"], "block": blk, "dtype": "float64",
                              "threads": 1, "kind": kind, "rep_index": int(idx),
                              "elapsed_ns": int(ns), "batch": int(b),
                              "time_per_mult_s": int(ns) / 1e9 / int(b), "checksum": chk,
                              "nnz_out": nnz, "status": status})
            runs.writerow({**base, **meta, "status": status, "exit_code": code, "batch": batch,
                           "nnz_out": nnz_out, "wall_time_s": round(wall, 3),
                           "command": " ".join(cmd)})
            raw_f.flush()
            runs_f.flush()
            print(f"{status} ({wall:.1f}s, peak RSS {meta['peak_rss_bytes'] / 2**20:.1f} MiB)")
            if status != "ok":
                f = (exp, m, blk, n, d)
                failures.append(f)
                failure_status[f] = status

    raw_f.close()
    runs_f.close()
    mins = (time.monotonic() - t_start) / 60
    print(f"\nFinished in {mins:.1f} min. Results written to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
