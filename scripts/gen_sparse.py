#!/usr/bin/env python3
"""Generate the sparse input matrices for Part B.

Structure : every one of the n*n positions is nonzero independently with
            probability `density` (Bernoulli / Erdos-Renyi pattern, uniform over
            the matrix, no banded or clustered structure). Positions are drawn
            with geometric gaps, which gives exactly the same distribution as
            one coin flip per position but costs O(nnz) instead of O(n^2).
Values    : uniform in [-1, 1) from Python's `random.Random` (Mersenne Twister),
            redrawn if exactly 0.0 so that every stored entry is a real nonzero.
Seed      : the string f"{seed}:{n}:{density}:{name}", deterministic across
            runs and platforms.
Density   : fraction of nonzero entries (sparsity = 1 - density). The REALISED
            nnz is recorded, because it differs slightly from density * n^2.

File format (little-endian), sorted row-major, no duplicates:
    int64 n, int64 nnz, int32 rows[nnz], int32 cols[nnz], float64 vals[nnz]
Files     : data/sparse/A_n{n}_d{density}.coo, data/sparse/B_n{n}_d{density}.coo
            (git-ignored, regenerable)
Manifest  : inputs/sparse_manifest.csv with nnz and SHA-256 of every file.
"""
import argparse
import csv
import hashlib
import math
import random
import struct
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sparse"
MANIFEST = ROOT / "inputs" / "sparse_manifest.csv"
LOW, HIGH = -1.0, 1.0


def dtag(density):
    """Stable text form of a density for file names, e.g. 0.001 -> '0.001'."""
    return f"{density:g}"


def paths(n, density):
    return (DATA / f"A_n{n}_d{dtag(density)}.coo", DATA / f"B_n{n}_d{dtag(density)}.coo")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def positions(rng, total, density):
    """Indices 0..total-1, each included independently with probability density."""
    if density <= 0:
        return
    if density >= 1:
        yield from range(total)
        return
    log_q = math.log1p(-density)
    p = -1
    while True:
        u = rng.random()                       # in [0, 1)
        p += 1 + int(math.log1p(-u) / log_q)   # geometric gap (failures before a success)
        if p >= total:
            return
        yield p


def write_coo(path, n, density, seed_str):
    rng = random.Random(seed_str)
    rows, cols, vals = array("i"), array("i"), array("d")
    for p in positions(rng, n * n, density):
        v = 0.0
        while v == 0.0:
            v = rng.uniform(LOW, HIGH)
        rows.append(p // n)
        cols.append(p % n)
        vals.append(v)
    if sys.byteorder != "little":
        for a in (rows, cols, vals):
            a.byteswap()
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", n, len(vals)))
        rows.tofile(f)
        cols.tofile(f)
        vals.tofile(f)
    return len(vals)


def read_coo(path):
    """Read a COO file back (used by the tests). Returns n, rows, cols, vals."""
    with open(path, "rb") as f:
        n, nnz = struct.unpack("<qq", f.read(16))
        rows, cols, vals = array("i"), array("i"), array("d")
        rows.fromfile(f, nnz)
        cols.fromfile(f, nnz)
        vals.fromfile(f, nnz)
    if sys.byteorder != "little":
        for a in (rows, cols, vals):
            a.byteswap()
    return n, rows, cols, vals


def write_coo_from_dense(path, n, dense_rows):
    """Write any dense matrix (list of rows) as COO; used by the tests for known cases."""
    rows, cols, vals = array("i"), array("i"), array("d")
    for i, r in enumerate(dense_rows):
        for j, v in enumerate(r):
            if v != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(float(v))
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", n, len(vals)))
        rows.tofile(f)
        cols.tofile(f)
        vals.tofile(f)


def read_manifest():
    rows = {}
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as f:
            for r in csv.DictReader(f):
                rows[(int(r["n"]), r["density"], r["matrix"])] = r
    return rows


def write_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields = ["n", "density", "matrix", "file", "seed_string", "nnz", "realised_density",
              "generator", "pattern", "distribution", "bytes", "sha256"]
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for key in sorted(rows, key=lambda k: (k[0], float(k[1]), k[2])):
            w.writerow(rows[key])


def ensure(n, density, seed):
    """Create A and B for (n, density) if missing; verify against the manifest if present."""
    DATA.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    for name, path in zip("AB", paths(n, density)):
        seed_str = f"{seed}:{n}:{dtag(density)}:{name}"
        if not path.exists():
            write_coo(path, n, density, seed_str)
        digest = sha256(path)
        key = (n, dtag(density), name)
        if key in rows and rows[key]["seed_string"] == seed_str and rows[key]["sha256"] != digest:
            sys.exit(f"Checksum mismatch for {path}: delete it to regenerate, or check the seed.")
        with open(path, "rb") as f:
            _, nnz = struct.unpack("<qq", f.read(16))
        rows[key] = {
            "n": n, "density": dtag(density), "matrix": name,
            "file": f"data/sparse/{path.name}", "seed_string": seed_str, "nnz": nnz,
            "realised_density": f"{nnz / (n * n):.6g}",
            "generator": "python random.Random (MT19937), geometric gaps",
            "pattern": "independent Bernoulli(density) per position",
            "distribution": f"uniform[{LOW},{HIGH}) excluding 0.0",
            "bytes": path.stat().st_size, "sha256": digest,
        }
    write_manifest(rows)
    return paths(n, density)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--densities", type=float, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=20262027)
    args = p.parse_args()
    for n in args.sizes:
        for d in args.densities:
            a, b = ensure(n, d, args.seed)
            print(f"n={n} density={dtag(d)}: {a.name}, {b.name}")


if __name__ == "__main__":
    main()
