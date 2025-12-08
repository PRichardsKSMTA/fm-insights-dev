#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
orchestrate.py
Runs the 3-step pipeline end-to-end:

1) summarize.py --csv <CSV>
2) narrate.py   --summary <summary.json> --model <model>
3) render_pdfs.py --client_json <client.json> --internal_json <internal.json> --outdir out

Fixes included:
- Always resolve the CSV path. If a bare filename is given (e.g., KIVI_20250830_OpenAIAPI.csv),
  we look in ./in/ for it. --latest scans ./in/ for newest CSV by date or mtime.
- Child process outputs are parsed to extract produced paths, with fallbacks to newest files.

Usage examples:
  python orchestrate.py --latest --model gpt-4.1
  python orchestrate.py --csv "in\\KIVI_20250830_OpenAIAPI.csv" --model gpt-4.1
"""

import argparse
import os
import re
import sys
import glob
import subprocess
from datetime import datetime
from typing import Optional, Tuple, List

IN_DIR = "in"
WORK_DIR = "work"
OUT_DIR = "out"

CSV_PATTERN = re.compile(r"^(?P<scac>[A-Za-z0-9]+)_(?P<date>\d{8})_OpenAIAPI\.csv$", re.IGNORECASE)

def log(msg: str) -> None:
    print(msg, flush=True)

def resolve_csv_path(user_value: Optional[str], use_latest: bool) -> Tuple[str, str, str]:
    """
    Returns (abs_csv_path, scac, yyyymmdd)
    If use_latest=True, find latest in IN_DIR by date in filename, else by mtime.
    If user provided a value, accept absolute path; otherwise search IN_DIR for basename.
    """
    if use_latest:
        csv_path, scac, ymd = find_latest_csv_in_in()
        if not csv_path:
            raise FileNotFoundError(f"No CSV files found in ./{IN_DIR}")
        return csv_path, scac, ymd

    if not user_value:
        raise ValueError("Either --csv or --latest is required.")

    # If absolute or relative path exists -> normalize to absolute
    if os.path.exists(user_value):
        base = os.path.basename(user_value)
        m = CSV_PATTERN.match(base)
        scac = m.group("scac") if m else (base.split("_")[0] if "_" in base else "CLIENT")
        ymd = m.group("date") if m else "00000000"
        return os.path.abspath(user_value), scac, ymd

    # If only a filename was given, look in IN_DIR
    candidate = os.path.join(IN_DIR, os.path.basename(user_value))
    if os.path.exists(candidate):
        base = os.path.basename(candidate)
        m = CSV_PATTERN.match(base)
        scac = m.group("scac") if m else (base.split("_")[0] if "_" in base else "CLIENT")
        ymd = m.group("date") if m else "00000000"
        return os.path.abspath(candidate), scac, ymd

    raise FileNotFoundError(f"CSV not found: {user_value} (also tried {candidate})")

def find_latest_csv_in_in() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (abs_path, scac, yyyymmdd) for the 'latest' CSV in IN_DIR.
    Primary: by YYYYMMDD in filename. Fallback: by mtime.
    """
    pattern = os.path.join(IN_DIR, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None, None, None

    dated: List[Tuple[datetime, str, str, str]] = []  # (dt, abs_path, scac, ymd)
    undated: List[Tuple[float, str]] = []             # (mtime, abs_path)

    for f in files:
        base = os.path.basename(f)
        m = CSV_PATTERN.match(base)
        if m:
            ymd = m.group("date")
            try:
                dt = datetime.strptime(ymd, "%Y%m%d")
            except Exception:
                dt = datetime.fromtimestamp(os.path.getmtime(f))
            dated.append((dt, os.path.abspath(f), m.group("scac"), ymd))
        else:
            undated.append((os.path.getmtime(f), os.path.abspath(f)))

    if dated:
        dated.sort(key=lambda t: t[0])
        _, path, scac, ymd = dated[-1]
        return path, scac, ymd

    # Fallback by mtime
    undated.sort(key=lambda t: t[0])
    path = undated[-1][1]
    base = os.path.basename(path)
    scac = base.split("_")[0] if "_" in base else "CLIENT"
    return path, scac, "00000000"

def run(cmd: list, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Run a subprocess, capture stdout/stderr as text.
    """
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def parse_summary_path(stdout: str) -> Optional[str]:
    """
    summarize.py prints the path to the summary JSON on its last line.
    Accept any line that ends with _summary.json or .json and exists.
    """
    cand = None
    for line in stdout.strip().splitlines():
        line = line.strip().strip('"')
        if line.lower().endswith("_summary.json") or line.lower().endswith(".json"):
            if os.path.exists(line):
                cand = line
    return cand

def newest_summary_path() -> Optional[str]:
    files = glob.glob(os.path.join(WORK_DIR, "*_summary.json"))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]

def parse_narrate_json_paths(stdout: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find client/internal JSONs in narrate.py stdout.
    """
    client = None
    internal = None
    for line in stdout.strip().splitlines():
        s = line.strip().strip('"')
        if s.lower().endswith("_client.json") and os.path.exists(s):
            client = s
        elif s.lower().endswith("_internal.json") and os.path.exists(s):
            internal = s
    return client, internal

def newest_reports() -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback to newest client/internal JSON in OUT_DIR.
    """
    clients = glob.glob(os.path.join(OUT_DIR, "*_client.json"))
    internals = glob.glob(os.path.join(OUT_DIR, "*_internal.json"))
    client = max(clients, key=os.path.getmtime) if clients else None
    internal = max(internals, key=os.path.getmtime) if internals else None
    return client, internal

def main():
    ap = argparse.ArgumentParser(description="Run summarize -> narrate -> render pipeline.")
    ap.add_argument("--csv", help="Path or filename of the CSV (we will also look in ./in/).")
    ap.add_argument("--latest", action="store_true", help="Use the latest CSV in ./in/")
    ap.add_argument("--model", default="gpt-4.1", help="Model for narrate.py")
    args = ap.parse_args()

    # Resolve CSV
    csv_path, scac, ymd = resolve_csv_path(args.csv, args.latest)
    log(f"Orchestrating for CSV: {os.path.basename(csv_path)}  (client={scac}, as-of={ymd})")

    # Step 1: Summarize
    log("\nStep 1: Summarize (CSV -> summary JSON)")
    py = sys.executable
    cmd1 = [py, "summarize.py", "--csv", csv_path]
    log(f"$ {py} summarize.py --csv {os.path.basename(csv_path)}")
    code, out, err = run(cmd1)
    if code != 0:
        log(err.strip() or out.strip())
        log("Step 1: Summarize (CSV -> summary JSON) failed with exit code " + str(code))
        sys.exit(1)

    summary_path = parse_summary_path(out) or newest_summary_path()
    if not summary_path or not os.path.exists(summary_path):
        log("ERROR: Could not determine summary JSON path from summarize.py output; looked in ./work as fallback.")
        sys.exit(1)
    log(f"Summary JSON: {summary_path}")

    # Step 2: Narrate
    log("\nStep 2: Narrate (summary JSON -> client/internal JSON)")
    cmd2 = [py, "narrate.py", "--summary", summary_path, "--model", args.model]
    log(f"$ {py} narrate.py --summary {os.path.basename(summary_path)} --model {args.model}")
    code, out, err = run(cmd2)
    if code != 0:
        log(err.strip() or out.strip())
        log("Step 2: Narrate failed with exit code " + str(code))
        sys.exit(1)

    client_json, internal_json = parse_narrate_json_paths(out)
    if not (client_json and internal_json):
        # fallback to newest pair
        cj, ij = newest_reports()
        client_json = client_json or cj
        internal_json = internal_json or ij
    if not (client_json and internal_json):
        log("ERROR: Could not determine client/internal JSONs from narrate.py output; looked in ./out as fallback.")
        sys.exit(1)
    log(f"Client JSON:   {client_json}")
    log(f"Internal JSON: {internal_json}")

    # Step 3: Render HTML
    log("\nStep 3: Render (JSON -> HTML)")
    cmd3 = [py, "render_pdfs.py", "--client_json", client_json, "--internal_json", internal_json, "--outdir", OUT_DIR]
    log(f"$ {py} render_pdfs.py --client_json {os.path.basename(client_json)} --internal_json {os.path.basename(internal_json)} --outdir {OUT_DIR}")
    code, out, err = run(cmd3)
    if code != 0:
        log(err.strip() or out.strip())
        log("Step 3: Render failed with exit code " + str(code))
        sys.exit(1)

    log(out.strip() or "Rendered.")

if __name__ == "__main__":
    main()
