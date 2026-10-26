#!/usr/bin/env python3
"""Generate the dense input matrices (reused unchanged from Assignment 1).

Copied from amaisuarez/assignment1, commit 754ec23, matmul-assignment1/scripts/gen_matrices.py.
Same seed and generator, so the files are byte-identical to the Assignment 1 inputs;
inputs/manifest.csv (also copied) holds their SHA-256 and the script refuses to
continue if a regenerated file does not match.

Original description: generate the shared input matrices used by ALL implementations.

Every language reads the same binary files, so the logical inputs are identical
across methods (cross-language random generators would otherwise differ).

Format : raw IEEE-754 float64, little-endian, row-major, n*n values, no header.
Values : uniform in [-1, 1) from Python's `random.Random` (Mersenne Twister).
Seed   : the string f"{seed}:{n}:{name}" (string seeding is deterministic
         across runs and platforms since Python 3.2).
Files  : data/A_n{n}.bin, data/B_n{n}.bin  (git-ignored, regenerable)
Manifest: inputs/manifest.csv with SHA-256 of every file (committed).
"""
import argparse
import csv
import hashlib
import random
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFEST = ROOT / "inputs" / "manifest.csv"
LOW, HIGH = -1.0, 1.0


def paths(n):
    return DATA / f"A_n{n}.bin", DATA / f"B_n{n}.bin"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_matrix(path, n, seed_str):
    rng = random.Random(seed_str)
    a = array("d", (rng.uniform(LOW, HIGH) for _ in range(n * n)))
    if sys.byteorder != "little":
        a.byteswap()
    with open(path, "wb") as f:
        a.tofile(f)


def read_manifest():
    rows = {}
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as f:
            for r in csv.DictReader(f):
                rows[(int(r["n"]), r["matrix"])] = r
    return rows


def write_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields = ["n", "matrix", "file", "seed_string", "generator", "distribution", "dtype", "bytes", "sha256"]
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


def ensure(n, seed):
    """Create A and B for size n if missing; verify against manifest if present."""
    DATA.mkdir(exist_ok=True)
    rows = read_manifest()
    for name, path in zip("AB", paths(n)):
        seed_str = f"{seed}:{n}:{name}"
        if not path.exists():
            write_matrix(path, n, seed_str)
        digest = sha256(path)
        key = (n, name)
        if key in rows and rows[key]["seed_string"] == seed_str and rows[key]["sha256"] != digest:
            sys.exit(f"Checksum mismatch for {path}: regenerate it (delete the file) or check the seed.")
        rows[key] = {
            "n": n, "matrix": name, "file": f"data/{path.name}", "seed_string": seed_str,
            "generator": "python random.Random (MT19937)", "distribution": f"uniform[{LOW},{HIGH})",
            "dtype": "float64-le", "bytes": path.stat().st_size, "sha256": digest,
        }
    write_manifest(rows)
    return paths(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=20262027)
    args = p.parse_args()
    for n in args.sizes:
        a, b = ensure(n, args.seed)
        print(f"n={n}: {a.name}, {b.name}")


if __name__ == "__main__":
    main()
