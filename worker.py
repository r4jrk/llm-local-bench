
import time
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def run_single(model, prompt, timeout):
    payload = {"model": model, "prompt": prompt, "stream": True}
    t0 = time.time()
    t_first = None
    token_times = []
    last = None

    try:
        r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "ttft": None,
            "wall_s": None,
            "prompt_tps": 0,
            "gen_tps": 0,
            "steady_tps": 0,
            "tokens": 0,
            "text": ""
        }

    full_text = ""

    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode())
        if t_first is None and "response" in data and data["response"].strip():
            t_first = time.time()
        if "response" in data:
            token_times.append(time.time())
            full_text += data["response"]
        last = data

    t1 = time.time()

    prompt_tokens = last.get("prompt_eval_count",0)
    prompt_duration = last.get("prompt_eval_duration",0)
    gen_tokens = last.get("eval_count",0)
    gen_duration = last.get("eval_duration",0)

    prompt_tps = (prompt_tokens/(prompt_duration/1e9)) if prompt_duration else 0
    gen_tps = (gen_tokens/(gen_duration/1e9)) if gen_duration else 0

    steady = 0
    if len(token_times)>20:
        t_start = token_times[20]
        t_end = token_times[-1]
        steady = (len(token_times)-20)/(t_end-t_start) if t_end>t_start else 0

    return {
        "ttft": (t_first - t0) if t_first else None,
        "wall_s": t1 - t0,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "steady_tps": steady,
        "tokens": gen_tokens,
        "text": full_text,
    }
