#!/usr/bin/env python3
"""Correctness checks for Assignment 2, run BEFORE any benchmark.

1. Known answers: identity, zero and a hand-computed 2x2, for every method.
2. Dense optimisations: ikj and blocked (several block sizes, including sizes that
   do not divide n) must be BIT-IDENTICAL to the Assignment 1 baseline, because
   they add the same products in the same k order (see java/Dense.java).
3. Reference library: the baseline is compared with NumPy's A @ B (reference
   only, not one of the studied methods) using the standard forward error bound
   of a dot product of length n (Higham, Accuracy and Stability of Numerical
   Algorithms, 2nd ed., Sec. 3.5):
       |C - C_exact| <= gamma_n * (|A| |B|),   gamma_n = n u / (1 - n u),  u = 2^-53
   Both results satisfy it, so we accept |C - C_numpy| <= 2 gamma_n (|A| |B|).
   This tolerance scales with n and with the magnitude of the terms, unlike a
   fixed rtol/atol.
4. Sparse CSR: on the same COO inputs, the sparse result must equal the dense
   baseline exactly (value by value), and its structural nnz must equal the nnz
   of the boolean product pattern(A) x pattern(B).
5. Generators: dense inputs are byte-identical to Assignment 1 (SHA-256 in
   inputs/manifest.csv); the realised sparse density is within 5 standard
   deviations of the target.

Usage: python3 tests/validate.py      (after ./build.sh)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_matrices  # noqa: E402
import gen_sparse  # noqa: E402

TMP = ROOT / "tests" / "tmp"
CP = str(ROOT / "java" / "build")
DENSE_METHODS = ["baseline", "ikj", "blocked"]
U = 2.0 ** -53
failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def write_dense(path, M):
    np.ascontiguousarray(M, dtype="<f8").tofile(path)


def run(method, n, pa, pb, fmt="dense", block=64):
    out = TMP / f"C_{method}_{n}_{block}.bin"
    cmd = ["java", "-cp", CP, "MatMul2", "--method", method, "--n", str(n), "--a", str(pa),
           "--b", str(pb), "--format", fmt, "--block", str(block), "--warmup", "0",
           "--reps", "1", "--batch", "1", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    meta = {}
    for line in r.stdout.splitlines():
        if line.startswith("#"):
            k, v = line[1:].split(",")
            meta[k] = int(v)
        elif line.startswith("measured"):
            meta["nnz_out"] = int(line.split(",")[5])
    return np.fromfile(out, dtype="<f8").reshape(n, n), meta


def known_cases():
    print("1. Known answers")
    cases = {
        "identity x M": (np.eye(4), np.arange(16, dtype=float).reshape(4, 4) - 7.5),
        "M x identity": (np.arange(16, dtype=float).reshape(4, 4) * 0.25, np.eye(4)),
        "zero x M": (np.zeros((3, 3)), np.arange(9, dtype=float).reshape(3, 3)),
        "2x2 by hand": (np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]])),
    }
    expected = {"2x2 by hand": np.array([[19., 22.], [43., 50.]])}
    for name, (A, B) in cases.items():
        n = A.shape[0]
        want = expected.get(name, A @ B)       # identity/zero cases are exact
        pa, pb = TMP / "ka.bin", TMP / "kb.bin"
        write_dense(pa, A)
        write_dense(pb, B)
        for m in DENSE_METHODS:
            for bs in ([64, 1, 3] if m == "blocked" else [64]):
                C, _ = run(m, n, pa, pb, block=bs)
                check(np.array_equal(C, want), f"{name}: {m}" + (f" bs={bs}" if m == "blocked" else ""))
        ca, cb = TMP / "ka.coo", TMP / "kb.coo"
        gen_sparse.write_coo_from_dense(ca, n, A.tolist())
        gen_sparse.write_coo_from_dense(cb, n, B.tolist())
        C, meta = run("sparse", n, ca, cb, fmt="coo")
        check(np.array_equal(C, want), f"{name}: sparse (nnz_out={meta['nnz_out']})")


def dense_random():
    print("2-3. Random dense inputs: optimised == baseline bit for bit; baseline vs NumPy")
    rng = np.random.default_rng(20262027)
    for n in [1, 2, 3, 7, 16, 37, 64, 100, 130]:
        A = rng.uniform(-1, 1, (n, n))
        B = rng.uniform(-1, 1, (n, n))
        pa, pb = TMP / "ra.bin", TMP / "rb.bin"
        write_dense(pa, A)
        write_dense(pb, B)
        base, _ = run("baseline", n, pa, pb)
        gamma = n * U / (1 - n * U)
        bound = 2 * gamma * (np.abs(A) @ np.abs(B))
        err = np.abs(base - A @ B)
        check(np.all(err <= bound), f"n={n:3d} baseline vs NumPy: max err {err.max():.2e}, "
                                    f"max allowed {bound.max():.2e}")
        C, _ = run("ikj", n, pa, pb)
        check(np.array_equal(C, base), f"n={n:3d} ikj == baseline (bitwise)")
        for bs in [7, 16, 64, 128]:
            C, _ = run("blocked", n, pa, pb, block=bs)
            check(np.array_equal(C, base), f"n={n:3d} blocked bs={bs} == baseline (bitwise)")


def sparse_random():
    print("4. Sparse CSR vs dense baseline on the same COO inputs")
    for n in [1, 5, 37, 100, 200]:
        for d in [0.0, 0.01, 0.05, 0.2, 0.5, 1.0]:
            ca, cb = TMP / "sa.coo", TMP / "sb.coo"
            nnz_a = gen_sparse.write_coo(ca, n, d, f"test:{n}:{d}:A")
            nnz_b = gen_sparse.write_coo(cb, n, d, f"test:{n}:{d}:B")
            Cs, ms = run("sparse", n, ca, cb, fmt="coo")
            Cd, _ = run("baseline", n, ca, cb, fmt="coo")
            Cb, _ = run("blocked", n, ca, cb, fmt="coo", block=16)
            _, ra, ca_, va = gen_sparse.read_coo(ca)
            _, rb, cb_, vb = gen_sparse.read_coo(cb)
            PA = np.zeros((n, n), dtype=np.int64)
            PB = np.zeros((n, n), dtype=np.int64)
            PA[np.array(ra, dtype=int), np.array(ca_, dtype=int)] = 1
            PB[np.array(rb, dtype=int), np.array(cb_, dtype=int)] = 1
            structural = int(((PA @ PB) > 0).sum())
            ok = (np.array_equal(Cs, Cd) and np.array_equal(Cb, Cd)
                  and ms["nnz_out"] == structural and ms["nnz_a"] == nnz_a and ms["nnz_b"] == nnz_b)
            check(ok, f"n={n:3d} density={d:<4}: nnz A/B={nnz_a}/{nnz_b}, nnz C={ms['nnz_out']} "
                      f"(pattern {structural}); sparse == baseline == blocked")


def generators():
    print("5. Input generators")
    for n in [16, 64]:
        try:
            gen_matrices.ensure(n, 20262027)   # exits on SHA-256 mismatch with A1 manifest
            check(True, f"dense n={n}: byte-identical to the Assignment 1 inputs (SHA-256)")
        except SystemExit as e:
            check(False, f"dense n={n}: {e}")
    n, d = 1000, 0.01
    nnz = gen_sparse.write_coo(TMP / "g.coo", n, d, "test:generator")
    mean, sd = d * n * n, (n * n * d * (1 - d)) ** 0.5
    check(abs(nnz - mean) <= 5 * sd, f"sparse n={n} d={d}: nnz={nnz}, expected {mean:.0f} +- {5 * sd:.0f}")
    _, r, c, _ = gen_sparse.read_coo(TMP / "g.coo")
    keys = np.array(r, dtype=np.int64) * n + np.array(c, dtype=np.int64)
    check(bool(np.all(np.diff(keys) > 0)), "sparse file sorted row-major without duplicates")


def main():
    if not (ROOT / "java" / "build" / "MatMul2.class").exists():
        sys.exit("Build first: ./build.sh")
    TMP.mkdir(parents=True, exist_ok=True)
    known_cases()
    dense_random()
    sparse_random()
    generators()
    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
