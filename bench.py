import time
import json
import csv
import threading
import requests
import psutil
import subprocess
from datetime import datetime
from typing import List, Dict

import system_monitor

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    # "qwen3.6:35b-a3b-q4_K_M",
    "fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:latest",
    # "qwen3-coder-next:latest",
    # "qwen3.6:27b-q4_K_M",
    # "qwen3.6:latest",
    # "qwen3-coder:latest",
    "hf.co/speakleash/Bielik-11B-v3.0-Instruct-GGUF:Q4_K_M",
    # "mistral:latest",
    # "deepseek-coder",
    # "gemma4:31b",
    # "gemma4:26b",
    # "gemma4:e4b",
]

PROMPT = """You are a senior backend engineer.

Design a rate limiting system for a Spring Boot SaaS integrating with KSeF API.
Requirements:
- handle IP+NIP rate limits
- include retry/backoff strategy
- propose queue architecture
- include pseudocode
- discuss tradeoffs

Answer in ~400–500 words.
"""

RUNS_PER_MODEL = 3
TIMEOUT = 3600
ARTIFACT_DIR = "llm-artifacts"


def get_gpu_usage():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        ).decode().strip()
        gpu_util, mem = result.split(",")
        return float(gpu_util), float(mem)
    except Exception:
        return 0.0, 0.0


    def summary(self):
        if not self.samples:
            return {}
        
        def avg(key):
            return sum(s[key] for s in self.samples) / len(self.samples)

        def peak(key):
            return max(s[key] for s in self.samples)

        return {
            "cpu_avg": avg("cpu"),
            "gpu_avg": avg("gpu"),
            "ram_avg": avg("ram"),
            "disk_read_avg": avg("disk_read_bps"),
            "disk_read_peak": peak("disk_read_bps"),
            "disk_write_avg": avg("disk_write_bps"),
        }


def run_stream(model: str, run_id: int) -> Dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": True
    }

    monitor = SystemMonitor()
    monitor.start()

    t_start = time.time()
    t_first_token = None

    response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=TIMEOUT)

    full_text = ""
    last_json = None

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode())

        if t_first_token is None:
            t_first_token = time.time()

        if "response" in data:
            full_text += data["response"]

        last_json = data

    t_end = time.time()
    monitor.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ARTIFACT_DIR}/{model.replace(':','_')}_run{run_id}_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_text)

    eval_count = last_json.get("eval_count", 0)
    eval_duration = last_json.get("eval_duration", 0)

    tps = 0
    if eval_duration > 0:
        tps = eval_count / (eval_duration / 1e9)

    metrics = monitor.summary()

    return {
        "model": model,
        "tokens": eval_count,
        "tps": tps,
        "wall_s": t_end - t_start,
        "ttft": (t_first_token - t_start) if t_first_token else None,
        **metrics,
    }


def benchmark_model(model: str):
    results = []

    # warmup
    _ = run_stream(model, run_id=0)

    for i in range(RUNS_PER_MODEL):
        print(f"   Run {i+1}/{RUNS_PER_MODEL}")
        res = run_stream(model, run_id=i+1)
        results.append(res)

    # average
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals)

    return {
        "model": model,
        "tps": avg("tps"),
        "wall_s": avg("wall_s"),
        "ttft": avg("ttft"),
        "cpu_avg": avg("cpu_avg"),
        "gpu_avg": avg("gpu_avg"),
        "ram_avg": avg("ram_avg"),
    }


def main():
    import os
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    all_results = []

    for model in MODELS:
        print(f"\n=== {model} ===")
        try:
            res = benchmark_model(model)
            all_results.append(res)
        except Exception as e:
            print(f"ERROR: {e}")

    # sort by tps
    all_results.sort(key=lambda x: x["tps"], reverse=True)

    # print
    print("\n=== RESULTS ===")
    print(f"{'Model':28} | t/s | TTFT | Wall | CPU | GPU | RAM")
    print("-" * 80)

    for r in all_results:
        print(
            f"{r['model'][:28]:28} | "
            f"{r['tps']:5.1f} | "
            f"{r['ttft']:5.2f}s | "
            f"{r['wall_s']:5.2f}s | "
            f"{r['cpu_avg']:4.1f}% | "
            f"{r['gpu_avg']:4.1f}% | "
            f"{r['ram_avg']:4.1f}%"
        )

    # CSV export
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print("\nSaved results to benchmark_results.csv and artifacts/ directory")


if __name__ == "__main__":
    main()