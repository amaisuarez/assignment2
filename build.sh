#!/usr/bin/env bash
# Compiles the Java implementations with the fixed settings used in the study.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p java/build
echo "[build] javac --release 17"
javac --release 17 -Xlint:all -d java/build java/*.java
echo "[build] done"
