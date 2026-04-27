
def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    k = int(len(values)*p/100)
    return values[min(k,len(values)-1)]

def aggregate(results):
    def c(k): return [r[k] for r in results if r.get(k) is not None]

    return {
        "count": len(results),
        "ttft_p50": percentile(c("ttft"),50),
        "steady_tps_avg": sum(c("steady_tps"))/len(results),
    }

def score_quality(results):
    # naive proxy: longer + slower slightly penalized
    tokens = sum(r["tokens"] for r in results)/len(results)
    tps = sum(r["steady_tps"] for r in results)/len(results)
    return tokens * 0.01 + tps * 0.1
