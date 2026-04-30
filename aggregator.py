def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)

    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)

    # linear interpolation
    return values[f] + (values[c] - values[f]) * (k - f)


def aggregate(results):
    def collect(key):
        return [r[key] for r in results if r.get(key) is not None]

    ttft_vals = collect("ttft")
    tps_vals = collect("steady_tps")

    return {
        "count": len(results),

        # latency
        "ttft_p50": percentile(ttft_vals, 50),
        "ttft_p90": percentile(ttft_vals, 90),

        # throughput
        "steady_tps_avg": sum(tps_vals) / len(tps_vals) if tps_vals else 0,
        "steady_tps_p90": percentile(tps_vals, 90),

        # token stats (safe)
        "tokens_avg": (
            sum(r.get("tokens", 0) for r in results) / len(results)
            if results else 0
        ),
    }


def score_quality(results):
    """
    Deprecated heuristic quality score.
    Kept only as a fallback if judge is disabled.

    NOTE:
    Do NOT mix performance (TPS) into quality scoring.
    """
    if not results:
        return 0

    tokens = sum(r.get("tokens", 0) for r in results) / len(results)

    # very weak proxy: longer answers tend to be more complete
    return tokens * 0.01