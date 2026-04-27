import argparse
import time
import re
import json
from queue import Queue
import threading
import csv
import os
import requests

from worker import run_single
from aggregator import aggregate, score_quality
from system_monitor import SystemMonitor

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    # "qwen3.6:35b-a3b-q4_K_M",
    # "fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:latest",
    # "qwen3-coder-next:latest",
    # "qwen3.6:27b-q4_K_M",
    # "qwen3.6:latest",
    # "qwen3-coder:latest",
    # "hf.co/speakleash/Bielik-11B-v3.0-Instruct-GGUF:Q4_K_M",
    "mistral:latest",
    # "deepseek-coder",
    # "gemma4:31b",
    # "gemma4:26b",
    # "gemma4:e4b",
]

PROMPT = """You are a senior backend engineer.
Design a rate limiting system for a Spring Boot SaaS integrating with KSeF API.
Include pseudocode and tradeoffs."""

TIMEOUT = 600
ARTIFACT_DIR = "artifacts"
WARMUP_RUNS = 1
COOLDOWN_SECONDS = 3


def check_ollama_available(url, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url.replace("/api/generate", "/api/tags"), timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass

        print(f"Ollama not reachable (attempt {attempt+1}/{retries})...")
        time.sleep(delay)

    return False

def worker_thread(q, results, model):
    while True:
        item = q.get()
        if item is None:
            break
        results.append(run_single(model, PROMPT, TIMEOUT))
        q.task_done()

def run_concurrent(model, requests, concurrency):
    q = Queue()
    results = []
    threads = []
    for _ in range(concurrency):
        t = threading.Thread(target=worker_thread, args=(q, results, model))
        t.start()
        threads.append(t)

    for _ in range(requests):
        q.put(1)

    q.join()

    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()

    return results

def run_sequential(model, requests):
    results = []
    for i in range(requests):
        print(f"   Request {i+1}/{requests}")
        results.append(run_single(model, PROMPT, TIMEOUT))
    return results

def run_all(mode, requests, concurrency):
    all_results = []

    for model in MODELS:
        print(f"\n----- {model} -----")

        try:
            for _ in range(WARMUP_RUNS):
                warm = run_single(model, PROMPT, TIMEOUT)
                if warm.get("error"):
                    raise RuntimeError(f"Warmup failed: {warm['error']}")

            time.sleep(COOLDOWN_SECONDS)

            monitor = SystemMonitor()
            monitor.start()

            if mode == "sequential":
                results = run_sequential(model, requests)
            else:
                results = run_concurrent(model, requests, concurrency)

            monitor.stop()
            time.sleep(0.2)
            sys_metrics = monitor.summary()

            if not results:
                print(f"⚠️ No results for {model}")
                continue

            valid = [r for r in results if not r.get("error")]
            failed = len(results) - len(valid)

            if failed > 0:
                print(f"⚠️ {failed}/{len(results)} failed runs for {model}")

            if not valid:
                print(f"❌ All runs failed for {model}")
                continue

            stats = aggregate(valid)
            stats["model"] = model
            stats["quality_score"] = score_quality(valid)

            stats["tokens_per_request"] = (
                sum(r["tokens"] for r in valid) / len(valid)
            )

            stats["failed_runs"] = failed
            stats["total_runs"] = len(results)

            stats.update(sys_metrics)
            all_results.append(stats)
            save_artifacts(model, results, stats)

        except Exception as e:
            print(f"❌ ERROR in model {model}: {e}")

        finally:
            try:
                import requests as rq
                rq.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "keep_alive": 0},
                    timeout=2
                )
            except Exception:
                pass

            time.sleep(COOLDOWN_SECONDS)

    return all_results


def safe_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)


def save_artifacts(model, results, stats):
    model_dir = os.path.join("artifacts", safe_name(model))
    os.makedirs(model_dir, exist_ok=True)

    for i, r in enumerate(results, 1):
        with open(os.path.join(model_dir, f"run_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(r.get("text", ""))

        with open(os.path.join(model_dir, f"run_{i}.json"), "w") as f:
            json.dump(r, f, indent=2)

    with open(os.path.join(model_dir, "prompt.txt"), "w") as f:
        f.write(PROMPT)

    with open(os.path.join(model_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(model_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {model}\\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\\n")


def save_global_results(all_results):
    os.makedirs("results", exist_ok=True)

    with open("results/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open("results/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    sorted_results = sorted(all_results, key=composite_score, reverse=True)

    with open("results/leaderboard.txt", "w") as f:
        f.write(f"{'Model':28} | TPS | TTFT | Quality | Tokens\n")
        f.write("-" * 80 + "\n")

        for r in sorted_results:
            f.write(
                f"{r['model'][:28]:28} | "
                f"{r['steady_tps_avg']:.1f} | "
                f"{r['ttft_p50']:.2f} | "
                f"{r['quality_score']:.2f} | "
                f"{r['tokens_per_request']:.1f}\n"
            )


def composite_score(r):
    return (
        r["steady_tps_avg"] * 0.5 +
        r["quality_score"] * 0.3 -
        r["ttft_p50"] * 0.2
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequential","concurrent"], default="sequential")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    if not check_ollama_available(OLLAMA_URL):
        print("\n❌ Ollama is not running or not reachable at:", OLLAMA_URL)
        print("👉 Start it with: ollama serve")
        return

    results = run_all(args.mode, args.requests, args.concurrency)
    save_global_results(results)

    results.sort(key=lambda x: x["steady_tps_avg"], reverse=True)

    print("\n=== FINAL ===")
    print(f"{'Model':28} | TPS | TTFT | Quality | Tokens")
    for r in results:
        print(f"{r['model'][:28]:28} | {r['steady_tps_avg']:.1f} | {r['ttft_p50']:.2f} | {r['quality_score']:.2f} | {r['tokens_per_request']:.1f}")

    with open("results.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
