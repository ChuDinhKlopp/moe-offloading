#!/usr/bin/env python3
import json
import csv
from pathlib import Path

BASE_DIR = Path("./log")
BENCH_DIR = BASE_DIR / "bench"
OBS_DIR = BASE_DIR / "observability"

# --- Helpers -------------------------------------------------------------

def config_key(model, input_len, output_len, tp_size, dp_size, concurrency):
    """Key to join bench and observability."""
    return (model, int(input_len), int(output_len), int(tp_size), int(dp_size), int(concurrency))

def parse_obs_filename(path: Path):
    """
    obs_in4096_out1024_gpt-oss-120b_tp1_dp4_ep1_off2_con8.csv
    -> model, in_len, out_len, tp, dp, con
    """
    stem = path.stem  # without .csv
    parts = stem.split("_")
    # ['obs', 'in4096', 'out1024', 'gpt-oss-120b', 'tp1', 'dp4', 'ep1', 'off2', 'con8']
    assert parts[0] == "obs"
    in_len = int(parts[1][2:])   # drop 'in'
    out_len = int(parts[2][3:])  # drop 'out'
    model = parts[3]
    tp_size = int(parts[4][2:])  # drop 'tp'
    dp_size = int(parts[5][2:])  # drop 'dp'
    # ep = parts[6]  # not used
    # off = parts[7] # not used
    concurrency = int(parts[8][3:])  # drop 'con'
    return model, in_len, out_len, tp_size, dp_size, concurrency

def parse_bench_dirname(path: Path):
    """
    report_in4096_out1024_gpt-oss-120b_tp1_dp4_ep1_off2_con8
    -> model, in_len, out_len, tp, dp, con
    """
    name = path.name
    parts = name.split("_")
    # ['report', 'in4096', 'out1024', 'gpt-oss-120b', 'tp1', 'dp4', 'ep1', 'off2', 'con8']
    assert parts[0] == "report"
    in_len = int(parts[1][2:])
    out_len = int(parts[2][3:])
    model = parts[3]
    tp_size = int(parts[4][2:])
    dp_size = int(parts[5][2:])
    concurrency = int(parts[8][3:])
    return model, in_len, out_len, tp_size, dp_size, concurrency

def read_preemptions_from_csv(path: Path) -> int | None:
    """
    CSV:
      metric,val
      sum(vllm:num_preemptions_total),96
    Return the 'val' for the preemptions metric (or None if missing).
    """
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["metric"] == "sum(vllm:num_preemptions_total)":
                try:
                    return int(float(row["val"]))
                except ValueError:
                    return None
    return None

# --- Build observability map ---------------------------------------------

obs_map = {}  # config_key -> num_total_preemption

for csv_path in OBS_DIR.glob("obs_*.csv"):
    model, in_len, out_len, tp, dp, con = parse_obs_filename(csv_path)
    key = config_key(model, in_len, out_len, tp, dp, con)
    preemptions = read_preemptions_from_csv(csv_path)
    obs_map[key] = preemptions

# --- Walk bench reports and join -----------------------------------------

rows = []

for report_dir in BENCH_DIR.glob("report_*"):
    if not report_dir.is_dir():
        continue

    model, in_len, out_len, tp, dp, con = parse_bench_dirname(report_dir)
    key = config_key(model, in_len, out_len, tp, dp, con)
    num_preempt = obs_map.get(key)

    # Each dir should have exactly one JSON benchmark file
    json_files = list(report_dir.glob("*.json"))
    if not json_files:
        # Skip if no json
        continue
    json_path = json_files[0]

    with json_path.open() as f:
        data = json.load(f)

    row = {
        "Model": model,
        "Input Len": in_len,
        "Output Len": out_len,
        "In/Out Size": f"({in_len}, {out_len})",
        "Max Concurrency": con,
        "TP Size": tp,
        "DP Size": dp,
        "Parallel Config": f"({tp}, {dp})",
        "End-To-End  Latency (s)": data.get("duration"),
        "Request Throughput (req/s)": data.get("request_throughput"),
        "Output Token Throughput (tok/s)": data.get("output_throughput"),
        "Total Token Throughput (tok/s)": data.get("total_token_throughput"),
        "mean TTFT (ms)": data.get("mean_ttft_ms"),
        "mean TPOT (ms)": data.get("mean_tpot_ms"),
        "mean ITL (ms)": data.get("mean_itl_ms"),
        "num_total_preemption (reqs)": num_preempt,
    }

    rows.append(row)

# --- Write final CSV -----------------------------------------------------

output_path = BASE_DIR / "summary_report.csv"

fieldnames = [
    "Model",
    "Input Len",
    "Output Len",
    "In/Out Size",
    "Max Concurrency",
    "TP Size",
    "DP Size",
    "Parallel Config",
    "End-To-End  Latency (s)",
    "Request Throughput (req/s)",
    "Output Token Throughput (tok/s)",
    "Total Token Throughput (tok/s)",
    "mean TTFT (ms)",
    "mean TPOT (ms)",
    "mean ITL (ms)",
    "num_total_preemption (reqs)",
]

with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote {len(rows)} rows to {output_path}")

