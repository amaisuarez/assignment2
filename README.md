# Optimised and sparse matrix multiplication in Java

**Big Data – Individual Assignment 2** · Amai Suárez Navarro · Grado en Ciencia e Ingeniería de Datos (ULPGC), 2026-2027

This folder has everything behind my second Big Data assignment. It continues
Assignment 1 (repository https://github.com/amaisuarez/assignment1) and asks two questions:

> **A.** How much do loop reordering and cache blocking speed up the basic triple-loop
> multiplication in Java, and how does the gain change with the matrix size *n*?
>
> **B.** On the same inputs, for which sparsity levels does a sparse (CSR)
> multiplication beat a dense one, in time and in memory?

The full write-up is in `Assignment02_SuarezNavarro_Report.pdf`.

**Repository:** https://github.com/amaisuarez/assignment2

---

## What I found (short version)

All numbers come from my laptop, an Intel Core i5-5350U MacBook Air with 8 GB of RAM
(run `results/run_20261026-153110/`).

- **Loop reordering (i-k-j) is the big win.** For n ≥ 1024 it makes the Assignment 1
  multiplication 10–15× faster: one multiply-add goes from about 13 ns to about 1 ns.
  At n = 2048 the baseline didn't finish one multiplication within my 60 s limit; i-k-j
  took 7.7 s.
- **Cache blocking didn't add a measurable gain on top.** Blocked and i-k-j differed by
  −8 % to +25 %, which is the same size as the difference between two separate runs of the
  same configuration (up to 1.28×). The block size barely mattered between 16 and 512.
- **Auto-vectorisation wasn't the reason either.** Turning it off only made a clear
  difference at n = 1024; the gain comes from reading memory contiguously.
- **Sparse (CSR) beats dense up to about 35–50 % density** on random matrices: from
  28 000–32 000× faster at 0.1 % to 1.8–2.2× at 35 %, and equal or slightly slower at
  50 %. The reason is that one sparse product costs about 4 dense multiply-adds, so sparse
  wins while d² < 1/4.
- **The product fills in fast.** At 5 % input density C is already 92–99 % nonzero, so
  storing C as CSR takes more memory than dense from there on. The COO→CSR conversion only
  matters for very sparse inputs, and even then sparse stays hundreds of times faster.

All methods give **bitwise-identical results** to the Assignment 1 baseline.

---

## What I reuse from Assignment 1

| Reused | From ([`amaisuarez/assignment1`](https://github.com/amaisuarez/assignment1), commit `754ec23`) |
|---|---|
| Baseline kernel | `matmul-assignment1/java/MatMul.java`, method `matmul`, copied unchanged into `java/Dense.java` (`Dense.baseline`) |
| Dense inputs | `matmul-assignment1/scripts/gen_matrices.py` and `inputs/manifest.csv`, copied unchanged. Same seed `20262027`, so the files are byte-identical to the Assignment 1 inputs (the script checks the SHA-256) |
| Environment info | `matmul-assignment1/scripts/collect_env.py` (C compiler fields removed) |
| JVM settings | `-XX:+UseSerialGC -Xms256m -Xmx4g`, compiled with `--release 17` |

I chose **Java** for Assignment 2 because in Assignment 1 it was as fast as C for large
matrices, its results were identical to C and Python bit for bit, and it is the language I
will use for the parallel (A3) and distributed (A4) assignments.

---

## The methods

| Method | Part | What changes | File |
|---|---|---|---|
| `baseline` | A, ref. | Assignment 1 i-j-k triple loop. Inner loop walks down a column of B (stride *n*) | `java/Dense.java` |
| `ikj` | A | **Loop reordering**: loops *j* and *k* swapped. Inner loop reads a row of B and updates a row of C, both contiguous | `java/Dense.java` |
| `blocked` | A, B | **Cache blocking** on top of i-k-j: works on 64×64 tiles so a tile of A, B and C (96 KiB) is reused while it is in cache | `java/Dense.java` |
| `ikj_novec`, `blocked_novec` | A (aux.) | Same kernels with the JIT's auto-vectorisation turned off (`-XX:-UseSuperWord`). Not extra optimisations: they separate the memory-access effect from SIMD code generation | — |
| `sparse` | B | **CSR × CSR → CSR** with Gustavson's row-by-row algorithm (symbolic + numeric pass, dense accumulator). My own code, no sparse library | `java/Csr.java` |

All three dense kernels add the products for each C[i][j] in the same order (k = 0…n−1),
and Java does not reorder or fuse floating-point operations, so they give **bit-identical
results**. The sparse kernel also matches the dense baseline value by value on the same
inputs. `tests/validate.py` checks all of this.

---

## What's in the folder

```
java/Dense.java            dense kernels: baseline (A1), ikj, blocked
java/Csr.java              CSR format, COO->CSR conversion, sparse multiplication
java/MatMul2.java          benchmark driver (one process = one configuration)

scripts/gen_matrices.py    dense inputs (reused from A1, same files)
scripts/gen_sparse.py      sparse inputs for Part B (COO files, several densities)
scripts/run_benchmarks.py  runs the whole study
scripts/collect_env.py     saves info about the machine (CPU, RAM, versions...)
scripts/analyze.py         builds the tables and plots from the raw data
tests/validate.py          correctness checks (run before benchmarking)

config.json                settings of the real study (sizes, densities, repetitions, limits)
config_quick.json          1-minute smoke test, just to check that everything works
build.sh                   compiles the Java code

inputs/manifest.csv        seeds and SHA-256 of the dense inputs (same as A1)
inputs/sparse_manifest.csv seeds, nnz and SHA-256 of the sparse inputs
results/                   raw data of the study (run_20261026-153110) and the validation report
report/                    report source (report.md), tables and figures generated by analyze.py
Assignment02_SuarezNavarro_Report.pdf   the report
```

---

## What you need

- macOS or Linux (on Windows, use WSL)
- Java JDK 17 or newer (I used OpenJDK 21)
- Python 3.9 or newer, with NumPy (only to check results) and Matplotlib (plots):

```bash
pip install -r requirements.txt
```

---

## How to run it

**1. Compile**

```bash
bash build.sh
```

**2. Check correctness** (about 1 minute)

```bash
python3 tests/validate.py | tee results/validation_report.txt
```

It must finish with `ALL TESTS PASSED`.

**3. (Optional) Quick test, about 1 minute**

```bash
python3 scripts/run_benchmarks.py --config config_quick.json
```

**4. The real study**

Keep the laptop plugged in and don't use it meanwhile. On a Mac, `caffeinate` stops it
from going to sleep:

```bash
caffeinate -i python3 scripts/run_benchmarks.py    # on Linux, drop "caffeinate -i"
```

You can also run the three experiments separately, e.g.
`python3 scripts/run_benchmarks.py --only part_a`, `--only block_sweep` or `--only part_b`.
At the end it prints the folder with the results, like `results/run_YYYYMMDD-HHMMSS`.

**5. Tables and plots**

```bash
python3 scripts/analyze.py results/run_YYYYMMDD-HHMMSS
```

This writes the tables to `report/tables/` and the plots to `report/figures/`.

The input matrices are generated automatically in `data/` (ignored by git because they are
large) and always come out the same thanks to fixed seeds; the manifests hold their SHA-256.

---

## How I made the comparisons fair

| | What I did |
|---|---|
| Inputs | Part A: the A1 dense files, values uniform in [−1, 1), seed `20262027`. Part B: one COO file per matrix, read by **both** the sparse and the dense method, so they multiply exactly the same matrices |
| Sparse inputs | Each position is nonzero independently with probability = density (uniform pattern, no structure); values uniform in [−1, 1) without exact zeros; realised nnz recorded |
| Densities (Part B) | 0.1 %, 0.5 %, 1 %, 5 %, 10 %, 25 %, 35 %, 50 % nonzeros (sparsity 99.9 % … 50 %), at n = 1024 and 2048 |
| Numbers | float64 everywhere, sequential (one application thread) |
| What I time | Only the multiplication. Reading files and building C's dense array are outside the timer. Zeroing C (ikj, blocked) and allocating the sparse output (its size is unknown in advance) are inside, because they are part of those algorithms. The COO→CSR conversion is **measured separately** and reported both apart and added to the sparse time |
| Timer | `System.nanoTime()` (monotonic); the result's checksum is printed after every repetition so the JIT cannot skip the work |
| Warm-up | Fresh JVM per configuration; 3 warm-up rounds for n ≤ 512 and 1 above, each lasting at least 100 ms. Not included in the statistics |
| Small matrices | Each repetition repeats the multiplication until it lasts about 100 ms and reports the time per multiplication |
| Repetitions | 5 measured repetitions per configuration; median and interquartile range (IQR) |
| Order | Method order shuffled (fixed seed) for every size, to avoid drift bias |
| Memory | Analytical array storage of A, B and C (dense 8 bytes/element; CSR 12 bytes/nonzero + row pointers), plus peak RSS of the process and Java heap in use |
| Limits (set beforehand) | 60 s per repetition and 4 GB of estimated array storage. A configuration that exceeds them is recorded as `timeout` / `skipped_memory_budget`, and larger sizes (or higher densities) are not attempted for that method |

---

## Raw data format

- `results/run_*/raw_measurements.csv`: one row per warm-up, measured repetition or
  conversion, with experiment, method, n, density, block size, numeric type, threads,
  kind, time in ns, batch, time per multiplication, checksum and output nnz.
- `results/run_*/runs.csv`: one row per configuration, with status, batch, nnz of A, B and
  C, array storage, peak RSS, heap, budgets and the exact command.
- `results/run_*/environment.json`: machine and version details.

---

## Limitations

Everything was measured on one old laptop, so absolute times are specific to it. The
JIT decides by itself whether to vectorise a loop; the `_novec` runs show how much of the
gain comes from that. Peak RSS mostly reflects how much heap the JVM reserved, not how
much data is alive, so for Part B I compare the analytical array storage. The sparse
inputs have a uniform random pattern; matrices with structure (bands, blocks, power-law
rows) can behave differently.

---

## Use of AI

I used Claude (Anthropic) to help design the experiment, explain the techniques, generate
the first versions of the code and scripts, and draft the report from my results. I ran all
the experiments on my own computer, and every number in the report comes from those runs.
