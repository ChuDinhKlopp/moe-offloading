import time
import csv
import argparse
from nvitop import Device
from datetime import datetime
import vllm.envs as envs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    ap.add_argument("--duration", type=float, default=None, help="Total duration in seconds (omit to run until killed)")
    ap.add_argument("--logfile", type=str, default="gpu_log.csv", help="Path to CSV file")
    args = ap.parse_args()
    return args

def monitor(interval=1.0, duration=None, logfile="gpu_log.csv"):
    devices = Device.all()

    with open(logfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "layer",
            "gpu_index",
            "gpu_util (%)",
            "pcie_tx_throughput (KiB/s)",
            "pcie_rx_throughput (KiB/s)",
            "mem_used (GB)",
            "mem_total (GB)",
        ])

        start_time = time.time()
        while True:
            now = datetime.now().isoformat(timespec="seconds")

            for device in devices:
                writer.writerow([
                    envs.STEP_NUM,
                    envs.LAYER_ID,
                    device.index,
                    device.gpu_utilization(),
                    device.pcie_tx_throughput(),
                    device.pcie_rx_throughput(),
                    device.memory_used_human(),
                    device.memory_total_human(),
                ])

            f.flush()

            if duration is not None and time.time() - start_time >= duration:
                break

            time.sleep(interval)

def main():
    args = parse_args()
    monitor(interval=args.interval, duration=args.duration, logfile=args.logfile)

if __name__ == "__main__":
    main()
