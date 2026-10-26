#!/usr/bin/env python3
"""Record the execution environment (hardware, OS, runtime versions).

Reused from Assignment 1 (commit 754ec23) with the C compiler fields removed."""
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # tool missing
        return f"unavailable ({e.__class__.__name__})"


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def cpu_info():
    info = {"logical_processors": os.cpu_count()}
    if sys.platform.startswith("linux"):
        cpuinfo = read("/proc/cpuinfo")
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        info["model"] = m.group(1).strip() if m else platform.processor()
        cores = set()
        phys = core = None
        for line in cpuinfo.splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif not line.strip() and core is not None:
                cores.add((phys, core))
                phys = core = None
        info["physical_cores"] = len(cores) or None
        info["lscpu"] = sh(["lscpu"])
        gov = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip()
        info["scaling_governor"] = gov or "unknown"
        info["affinity_cpus_available"] = sorted(os.sched_getaffinity(0))
    elif sys.platform == "darwin":
        info["model"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["physical_cores"] = int(sh(["sysctl", "-n", "hw.physicalcpu"]) or 0) or None
    else:
        info["model"] = platform.processor()
        info["physical_cores"] = None
    return info


def ram_bytes():
    if sys.platform.startswith("linux"):
        m = re.search(r"MemTotal:\s+(\d+) kB", read("/proc/meminfo"))
        return int(m.group(1)) * 1024 if m else None
    if sys.platform == "darwin":
        try:
            return int(sh(["sysctl", "-n", "hw.memsize"]))
        except ValueError:
            return None
    return None


def collect(config):
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": cpu_info(),
        "ram_bytes": ram_bytes(),
        "python": {"implementation": platform.python_implementation(),
                   "version": sys.version, "executable": sys.executable},
        "numpy_version_reference_only": numpy_version,
        "java_version": sh(["java", "-version"]),
        "javac_version": sh(["javac", "-version"]),
        "java_opts": config.get("java_opts"),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain"]).strip()) if Path(".git").exists() else None,
        "threads_note": ("All kernels are sequential (one application thread). The JVM additionally "
                         "runs internal threads (JIT compiler, SerialGC, housekeeping); no numerical "
                         "library with its own thread pool is used."),
    }


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    print(json.dumps(collect(cfg), indent=2))
