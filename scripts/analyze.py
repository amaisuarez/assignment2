#!/usr/bin/env python3
"""Tables and figures for the Assignment 2 report from one benchmark run.

Usage: python3 scripts/analyze.py [results/run_XXXX]   (default: latest run)

Statistics per configuration use the MEASURED repetitions only (warm-ups are kept
in the raw CSV but excluded): median, Q1, Q3, IQR. All times are per
multiplication (a repetition times `batch` multiplications).

Derived values
  ns/iter   median / n^3   (dense methods; one inner-loop iteration = 1 multiply-add)
  GFLOP/s   2 n^3 / median (dense methods; useful work, not counting skipped zeros)
  speedup   median(baseline) / median(method), same n           (Part A)
  sparse vs dense = median(dense) / median(sparse), same n and density (Part B);
            also with the median COO->CSR conversion time added to sparse.

Outputs: report/tables/*.csv|*.md and report/figures/*.png
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_sparse  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {   # colour, marker, linestyle  (marker/linestyle = secondary encoding)
    "baseline": (BLUE, "o", "-"),
    "ikj": (ORANGE, "s", "-"),
    "blocked": (AQUA, "^", "-"),
    "ikj_novec": (ORANGE, "s", "--"),
    "blocked_novec": (AQUA, "^", "--"),
    "sparse": (BLUE, "o", "-"),
}
LABEL = {"baseline": "Baseline i-j-k (A1)", "ikj": "Loop reorder i-k-j",
         "blocked": "Cache blocking", "ikj_novec": "i-k-j, no auto-vectorisation",
         "blocked_novec": "Blocking, no auto-vectorisation", "sparse": "Sparse CSR"}
A_ORDER = ["baseline", "ikj", "blocked", "ikj_novec", "blocked_novec"]


def latest_run():
    runs = sorted((ROOT / "results").glob("run_*"))
    if not runs:
        sys.exit("No runs found in results/")
    return runs[-1]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_table(rows, cols, name, tables):
    with open(tables / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (tables / f"{name}.md").write_text("\n".join(lines) + "\n")


def fmt_time(s):
    if s is None:
        return "–"
    if s < 1e-3:
        return f"{s * 1e6:.2f} µs"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def stats(v):
    v = np.asarray(v)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": med, "q1": q1, "q3": q3, "iqr": q3 - q1, "count": len(v)}


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def plot_line(ax, x, y, method, label=None, yerr=None):
    c, mk, ls = STYLE[method]
    ax.errorbar(x, y, yerr=yerr, color=c, marker=mk, linestyle=ls, linewidth=2,
                markersize=6, markeredgecolor="white", markeredgewidth=1, capsize=0,
                label=label or LABEL[method])


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    raw = load_csv(run / "raw_measurements.csv")
    runs = load_csv(run / "runs.csv")
    env = json.loads((run / "environment.json").read_text())
    tables = ROOT / "report" / "tables"
    figs = ROOT / "report" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

    samples, convert = defaultdict(list), defaultdict(list)
    checks = defaultdict(set)
    for r in raw:
        if r["status"] != "ok":
            continue
        key = (r["experiment"], r["method"], int(r["n"]), r["density"], int(r["block"]))
        if r["kind"] == "measured":
            samples[key].append(float(r["time_per_mult_s"]))
            if r["experiment"] == "part_a":
                checks[int(r["n"])].add(r["checksum"])
        elif r["kind"] == "convert":
            convert[key].append(float(r["time_per_mult_s"]))
    info = {(r["experiment"], r["method"], int(r["n"]), r["density"], int(r["block"])): r for r in runs}
    S = {k: stats(v) for k, v in samples.items()}

    def mib(x):
        return f"{int(x) / 2**20:.1f}" if x not in ("", None) else "–"

    # ===================================================== Part A ============
    a_keys = [k for k in info if k[0] == "part_a"]
    if a_keys:
        block = a_keys[0][4]
        sizes = sorted({k[2] for k in a_keys})
        methods = [m for m in A_ORDER if any(k[1] == m for k in a_keys)]
        rows = []
        for m in methods:
            for n in sizes:
                k = ("part_a", m, n, "", block)
                if k not in info:
                    continue
                st, base = S.get(k), S.get(("part_a", "baseline", n, "", block))
                row = {"Method": m, "n": n, "Status": info[k]["status"]}
                if st:
                    row.update({
                        "Reps": st["count"], "Batch": info[k]["batch"],
                        "Median": fmt_time(st["median"]), "IQR": fmt_time(st["iqr"]),
                        "IQR/median": f"{100 * st['iqr'] / st['median']:.1f}%",
                        "ns/iter": f"{st['median'] / n**3 * 1e9:.3f}",
                        "GFLOP/s": f"{2 * n**3 / st['median'] / 1e9:.3f}",
                        "Speedup vs baseline": f"{base['median'] / st['median']:.2f}" if base else "–",
                        "Arrays (MiB)": mib(info[k]["array_storage_bytes"]),
                        "Peak RSS (MiB)": mib(info[k]["peak_rss_bytes"]),
                        "median_s": f"{st['median']:.6e}", "iqr_s": f"{st['iqr']:.6e}",
                    })
                rows.append(row)
        write_table(rows, ["Method", "n", "Status", "Reps", "Batch", "Median", "IQR", "IQR/median",
                           "ns/iter", "GFLOP/s", "Speedup vs baseline", "Arrays (MiB)",
                           "Peak RSS (MiB)", "median_s", "iqr_s"], "part_a_summary", tables)

        # limits
        lim = []
        for m in methods:
            ok = [n for n in sizes if info.get(("part_a", m, n, "", block), {}).get("status") == "ok"]
            fail = [(n, info[("part_a", m, n, "", block)]["status"]) for n in sizes
                    if ("part_a", m, n, "", block) in info
                    and info[("part_a", m, n, "", block)]["status"] != "ok"]
            lim.append({"Method": m, "Largest n completed": max(ok) if ok else "–",
                        "Failures / not attempted": "; ".join(f"n={n}: {s}" for n, s in fail) or "none"})
        write_table(lim, ["Method", "Largest n completed", "Failures / not attempted"], "part_a_limits", tables)

        # identical results check (bitwise identical C -> identical checksum string)
        ck = [{"n": n, "Distinct checksums across methods": len(checks[n]),
               "Checksum": next(iter(checks[n])) if len(checks[n]) == 1 else " / ".join(sorted(checks[n]))}
              for n in sorted(checks)]
        write_table(ck, ["n", "Distinct checksums across methods", "Checksum"], "part_a_checksums", tables)

        # figure 1: runtime
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for m in methods:
            xs = [n for n in sizes if ("part_a", m, n, "", block) in S]
            if not xs:
                continue
            ys = [S[("part_a", m, n, "", block)]["median"] for n in xs]
            err = [[S[("part_a", m, n, "", block)]["median"] - S[("part_a", m, n, "", block)]["q1"] for n in xs],
                   [S[("part_a", m, n, "", block)]["q3"] - S[("part_a", m, n, "", block)]["median"] for n in xs]]
            plot_line(ax, xs, ys, m, yerr=err)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        style_axes(ax, "Time per multiplication (median, IQR bars)", "Matrix size n (n × n)", "Seconds (log scale)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figs / "part_a_runtime.png")
        plt.close(fig)

        # figure 2: speedup
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for m in methods[1:]:
            xs = [n for n in sizes if ("part_a", m, n, "", block) in S and ("part_a", "baseline", n, "", block) in S]
            ys = [S[("part_a", "baseline", n, "", block)]["median"] / S[("part_a", m, n, "", block)]["median"] for n in xs]
            if xs:
                plot_line(ax, xs, ys, m)
        ax.axhline(1, color=INK2, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        style_axes(ax, "Speedup over the baseline (ratio of medians)", "Matrix size n", "× faster than baseline")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figs / "part_a_speedup.png")
        plt.close(fig)

        # figure 3: ns per inner iteration
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for m in methods:
            xs = [n for n in sizes if ("part_a", m, n, "", block) in S]
            if xs:
                plot_line(ax, xs, [S[("part_a", m, n, "", block)]["median"] / n**3 * 1e9 for n in xs], m)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        style_axes(ax, "Cost of one multiply-add (median time / n³)", "Matrix size n", "Nanoseconds (log scale)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figs / "part_a_ns_per_iter.png")
        plt.close(fig)

    # ================================================= Block sweep ===========
    b_keys = sorted(k for k in info if k[0] == "block_sweep")
    if b_keys:
        rows = []
        for n in sorted({k[2] for k in b_keys}):
            for k in sorted((k for k in b_keys if k[2] == n), key=lambda k: k[4]):
                st = S.get(k)
                rows.append({"n": n, "Block": k[4], "Tile memory 3·8·bs² (KiB)": f"{3 * 8 * k[4]**2 / 1024:g}",
                             "Status": info[k]["status"],
                             "Median": fmt_time(st["median"]) if st else "–",
                             "IQR": fmt_time(st["iqr"]) if st else "–",
                             "Reps": st["count"] if st else 0,
                             "median_s": f"{st['median']:.6e}" if st else ""})
        write_table(rows, ["n", "Block", "Tile memory 3·8·bs² (KiB)", "Status", "Reps", "Median", "IQR", "median_s"],
                    "block_sweep", tables)
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for n in sorted({k[2] for k in b_keys}):
            ks = [k for k in sorted(b_keys, key=lambda k: k[4]) if k[2] == n and k in S]
            xs = [k[4] for k in ks]
            ys = [S[k]["median"] for k in ks]
            err = [[S[k]["median"] - S[k]["q1"] for k in ks], [S[k]["q3"] - S[k]["median"] for k in ks]]
            plot_line(ax, xs, ys, "blocked", label=f"Cache blocking, n = {n}", yerr=err)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
        ax.set_xscale("log", base=2)
        ax.set_xticks([k[4] for k in b_keys])
        ax.set_xticklabels([str(k[4]) for k in b_keys])
        style_axes(ax, "Effect of the block size (median, IQR bars)", "Block size bs", "Seconds per multiplication")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figs / "block_sweep.png")
        plt.close(fig)

    # ===================================================== Part B ============
    p_keys = [k for k in info if k[0] == "part_b"]
    if p_keys:
        block = p_keys[0][4]
        dense_m = next(k[1] for k in p_keys if k[1] != "sparse")
        sizes = sorted({k[2] for k in p_keys})
        dens = sorted({k[3] for k in p_keys}, key=float)
        rows = []
        for n in sizes:
            for d in dens:
                ks, kd = ("part_b", "sparse", n, d, block), ("part_b", dense_m, n, d, block)
                if ks not in info and kd not in info:
                    continue
                s, dn = S.get(ks), S.get(kd)
                ri = info.get(ks) or info.get(kd)
                conv = stats(convert[ks])["median"] if convert.get(ks) else None
                nnz_c = info[ks]["nnz_out"] if ks in info and info[ks]["nnz_out"] else (
                    info[kd]["nnz_out"] if kd in info else "")
                row = {"n": n, "Density": d, "Sparsity": f"{100 * (1 - float(d)):g}%",
                       "nnz A": ri["nnz_a"], "nnz B": ri["nnz_b"], "nnz C": nnz_c,
                       "Output density": f"{int(nnz_c) / n**2:.4f}" if nnz_c != "" else "–",
                       "Sparse status": info.get(ks, {}).get("status", "–"),
                       "Sparse median": fmt_time(s["median"]) if s else "–",
                       "Sparse IQR": fmt_time(s["iqr"]) if s else "–",
                       "Conversion median": fmt_time(conv),
                       "Dense status": info.get(kd, {}).get("status", "–"),
                       "Dense median": fmt_time(dn["median"]) if dn else "–",
                       "Dense IQR": fmt_time(dn["iqr"]) if dn else "–",
                       "Dense/sparse": f"{dn['median'] / s['median']:.2f}" if s and dn else "–",
                       "Dense/(sparse+conv)": f"{dn['median'] / (s['median'] + conv):.2f}" if s and dn and conv else "–",
                       "Sparse arrays (MiB)": mib(info.get(ks, {}).get("array_storage_bytes")),
                       "Dense arrays (MiB)": mib(info.get(kd, {}).get("array_storage_bytes")),
                       "Sparse peak RSS (MiB)": mib(info.get(ks, {}).get("peak_rss_bytes")),
                       "Dense peak RSS (MiB)": mib(info.get(kd, {}).get("peak_rss_bytes")),
                       "sparse_median_s": f"{s['median']:.6e}" if s else "",
                       "dense_median_s": f"{dn['median']:.6e}" if dn else "",
                       "convert_median_s": f"{conv:.6e}" if conv else "",
                       "sparse_storage_bytes": info.get(ks, {}).get("array_storage_bytes", "")}
                rows.append(row)
        cols_time = ["n", "Density", "Sparsity", "nnz A", "nnz B", "nnz C", "Output density",
                     "Sparse median", "Sparse IQR", "Conversion median", "Dense median", "Dense IQR",
                     "Dense/sparse", "Dense/(sparse+conv)"]
        cols_mem = ["n", "Density", "nnz C", "Sparse arrays (MiB)", "Dense arrays (MiB)",
                    "Sparse peak RSS (MiB)", "Dense peak RSS (MiB)"]
        write_table(rows, cols_time + ["Sparse status", "Dense status"], "part_b_time", tables)
        write_table(rows, cols_mem, "part_b_memory", tables)
        write_table(rows, list(rows[0].keys()), "part_b_all", tables)

        # work of the sparse kernel: exact number of scalar products
        #   sum_k nnz(column k of A) * nnz(row k of B)   (expected n^3 d^2)
        # computed from the input files (regenerated from the seed if missing).
        seed = json.loads((run / "config.json").read_text())["seed"]
        work = []
        for n in sizes:
            for d in dens:
                ks, kd = ("part_b", "sparse", n, d, block), ("part_b", dense_m, n, d, block)
                if ks not in S:
                    continue
                pa, pb = gen_sparse.ensure(n, float(d), seed)
                _, _, ca, _ = gen_sparse.read_coo(pa)
                _, rb, _, _ = gen_sparse.read_coo(pb)
                col_a = np.bincount(np.asarray(ca, dtype=np.int64), minlength=n)
                row_b = np.bincount(np.asarray(rb, dtype=np.int64), minlength=n)
                prods = int((col_a * row_b).sum())
                row = {"n": n, "Density": d, "Products": prods,
                       "Expected n³d²": f"{n**3 * float(d)**2:.3g}",
                       "Sparse ns/product": f"{S[ks]['median'] / prods * 1e9:.2f}" if prods else "–",
                       "Dense ns/multiply-add": f"{S[kd]['median'] / n**3 * 1e9:.3f}" if kd in S else "–"}
                if prods and kd in S:
                    # density at which sparse and dense would take the same time if the
                    # cost per product stayed as measured here: d* = sqrt(dense / sparse cost)
                    row["Implied break-even density"] = f"{(S[kd]['median'] / n**3 / (S[ks]['median'] / prods)) ** 0.5:.2f}"
                work.append(row)
        write_table(work, ["n", "Density", "Products", "Expected n³d²", "Sparse ns/product",
                           "Dense ns/multiply-add", "Implied break-even density"], "part_b_work", tables)

        # figure: time vs density, one panel per n (shared y scale on each panel only)
        fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4.0), squeeze=False)
        for ax, n in zip(axes[0], sizes):
            for m, lab in [("sparse", "Sparse CSR (kernel)"), (dense_m, f"Dense ({LABEL[dense_m].lower()})")]:
                xs = [float(d) for d in dens if ("part_b", m, n, d, block) in S]
                ys = [S[("part_b", m, n, d, block)]["median"] for d in dens if ("part_b", m, n, d, block) in S]
                if xs:
                    plot_line(ax, xs, ys, m if m == "sparse" else "ikj", label=lab)
            xs = [float(d) for d in dens if ("part_b", "sparse", n, d, block) in S and convert.get(("part_b", "sparse", n, d, block))]
            ys = [S[("part_b", "sparse", n, d, block)]["median"] + stats(convert[("part_b", "sparse", n, d, block)])["median"]
                  for d in dens if ("part_b", "sparse", n, d, block) in S and convert.get(("part_b", "sparse", n, d, block))]
            if xs:
                ax.plot(xs, ys, color=BLUE, linestyle=":", linewidth=2, label="Sparse + COO→CSR conversion")
            ax.set_xscale("log")
            ax.set_yscale("log")
            style_axes(ax, f"n = {n}", "Input density (fraction of nonzeros, log)", "Seconds per multiplication (log)")
            ax.legend(frameon=False, fontsize=7.5)
        fig.suptitle("Sparse vs dense on the same inputs (medians)", x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(figs / "part_b_time_vs_density.png")
        plt.close(fig)

        # figure: fill-in of the product
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for n, (c, mk) in zip(sizes, [(BLUE, "o"), (ORANGE, "s"), (AQUA, "^")]):
            pts = [(float(r["Density"]), int(r["nnz C"]) / n**2) for r in rows if r["n"] == n and r["nnz C"] != ""]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=c, marker=mk, linewidth=2, markeredgecolor="white", label=f"Output C, n = {n}")
        allx = sorted({float(d) for d in dens})
        ax.plot(allx, allx, color=INK2, linestyle="--", linewidth=1, label="Same density as inputs")
        ax.set_xscale("log")
        ax.set_yscale("log")
        style_axes(ax, "Fill-in: the product is denser than its inputs", "Input density", "Output density")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figs / "part_b_fill_in.png")
        plt.close(fig)

        # figure: array storage
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for n, (c, mk) in zip(sizes, [(BLUE, "o"), (ORANGE, "s"), (AQUA, "^")]):
            pts = [(float(r["Density"]), int(r["sparse_storage_bytes"]) / 2**20) for r in rows
                   if r["n"] == n and r["sparse_storage_bytes"] not in ("", None)]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=c, marker=mk, linewidth=2, markeredgecolor="white", label=f"CSR A+B+C, n = {n}")
                ax.axhline(3 * 8 * n * n / 2**20, color=c, linestyle="--", linewidth=1.2,
                           label=f"Dense A+B+C, n = {n}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        style_axes(ax, "Array storage of A, B and C", "Input density", "MiB (log)")
        ax.legend(frameon=False, fontsize=7.5)
        fig.savefig(figs / "part_b_memory.png")
        plt.close(fig)

    # ============ same configuration in separate JVMs (process-to-process) ====
    same = defaultdict(list)
    for k, st in S.items():
        exp, m, n, d, blk = k
        if m == "blocked" and blk == 64:
            same[n].append((exp, d, st["median"]))
    rows = []
    for n, lst in sorted(same.items()):
        if len(lst) < 2:
            continue
        for exp, d, med in sorted(lst, key=lambda x: x[2]):
            rows.append({"n": n, "Experiment": exp, "Input density": d or "dense (A1 inputs)",
                         "Median": fmt_time(med), "median_s": f"{med:.6e}"})
        meds = [x[2] for x in lst]
        rows.append({"n": n, "Experiment": "ALL", "Input density": f"{len(lst)} processes",
                     "Median": f"range {fmt_time(min(meds))} – {fmt_time(max(meds))} "
                               f"(max/min {max(meds) / min(meds):.2f})", "median_s": ""})
    if rows:
        write_table(rows, ["n", "Experiment", "Input density", "Median", "median_s"],
                    "process_variation_blocked", tables)

    # environment summary
    cpu = env.get("cpu", {})
    (tables / "environment.md").write_text(
        f"- CPU: {cpu.get('model')} ({cpu.get('physical_cores')} physical cores, "
        f"{cpu.get('logical_processors')} logical processors)\n"
        f"- RAM: {(env.get('ram_bytes') or 0) / 1e9:.1f} GB\n"
        f"- OS: {env.get('os')}\n"
        f"- Java: {env.get('java_version', '').splitlines()[-2] if env.get('java_version') else ''}\n"
        f"- JVM options: {' '.join(env.get('java_opts') or [])}\n"
        f"- Python (harness/analysis): {env.get('python', {}).get('version', '').split()[0]}\n")
    print(f"Tables in {tables.relative_to(ROOT)}, figures in {figs.relative_to(ROOT)} (from {run.name})")


if __name__ == "__main__":
    main()
