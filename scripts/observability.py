import json
import argparse
import requests
from typing import List

def query_prometheus(metric: str):
    url = 'http://localhost:9090/api/v1/query'
    
    params = {'query': metric}

    try:
        response = requests.get(url, params=params)

        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] An error occured when querying from Prometheus: {e}")

    return response.json()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", type=str, help="Path to CSV file")
    args = ap.parse_args()
    return args


def parse_obs_json(metric: str, json_obj: str):
    # json_obj = json.loads(json_str)
    is_success = json_obj["status"]
    data = json_obj["data"]
    if not data["result"]:
        print("The result list is empty")
    else:
        val = data["result"][0]["value"][1]

        return {"metric": metric, "val": val}
    
import csv
import os

def log_dict(entry: dict, logfile: str) -> None:
    """
    Append a row to a CSV logfile.

    entry: {"metric": metric, "val": val}
    logfile: path to CSV file
    """
    # Check if file exists to know whether to write header
    file_exists = os.path.isfile(logfile)

    with open(logfile, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "val"])
        # Write header only once (if file is new/empty)
        if not file_exists or f.tell() == 0:
            writer.writeheader()
        writer.writerow(entry)


def main():
    args = parse_args()
    metric = 'sum(vllm:num_preemptions_total)'
    res = query_prometheus(metric)
    entry = parse_obs_json(metric, res)
    log_dict(entry, args.logfile)

if __name__ == "__main__":
    main()
