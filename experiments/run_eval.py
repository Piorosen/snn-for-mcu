#!/usr/bin/env python3
"""실험 E1~E3(+E5) 병렬 실행 래퍼.
eval_testset 바이너리를 W개 샤드로 병렬 실행해 집계하고 experiments/eval_results.json 저장.
사용: python3 experiments/run_eval.py [--n 10000] [--unfused-n 1000] [--workers 8]
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "host_tests", "bin", "eval_testset")
DATA = os.path.join(HERE, "testset_hwc.bin")

def run_shards(n, workers, extra=None):
    per = (n + workers - 1) // workers
    procs = []
    for w in range(workers):
        s, c = w * per, min(per, n - w * per)
        if c <= 0: break
        cmd = [BIN, DATA, str(s), str(c)] + (extra or [])
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True))
    agg = {}
    for p in procs:
        out, _ = p.communicate()
        assert p.returncode == 0, out
        line = [l for l in out.splitlines() if l.startswith("RESULT")][0]
        kv = dict(tok.split("=", 1) for tok in line.split()[1:])
        for k, v in kv.items():
            if k == "curve":
                cur = list(map(int, v.split(",")))
                agg.setdefault("curve", [0] * len(cur))
                agg["curve"] = [a + b for a, b in zip(agg["curve"], cur)]
            else:
                agg[k] = agg.get(k, 0) + int(v)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--unfused-n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    print(f"[E1-E3] fused pipeline, n={a.n}, workers={a.workers} ...", flush=True)
    r = run_shards(a.n, a.workers)
    n = r["n"]
    acc30 = r["correct30"] / n
    curve = [c / n for c in r["curve"]]
    dens = {
        "input": r["spikes_in"] / (r["steps"] * 3072),
        "lif1": r["spikes_l1"] / (r["steps"] * 8192),
        "lif2": r["spikes_l2"] / (r["steps"] * 4096),
    }
    print(f"  E1 integer accuracy (T=30): {acc30*100:.2f}% ({r['correct30']}/{n})")
    print(f"  E2 curve: T=5 {curve[4]*100:.2f}% | T=10 {curve[9]*100:.2f}% | T=15 {curve[14]*100:.2f}%"
          f" | T=20 {curve[19]*100:.2f}% | T=25 {curve[24]*100:.2f}% | T=30 {curve[29]*100:.2f}%")
    print(f"  E3 spike density: input {dens['input']*100:.1f}% | LIF1 {dens['lif1']*100:.1f}% | LIF2 {dens['lif2']*100:.1f}%")

    out = {"n": n, "acc_t30": acc30, "acc_curve_t1_30": curve, "spike_density": dens}

    if a.unfused_n > 0:
        print(f"[E5] fused vs unfused agreement, n={a.unfused_n} ...", flush=True)
        r2 = run_shards(a.unfused_n, a.workers, ["--unfused"])
        out["e5"] = {
            "n": r2["n"],
            "pred_agreement": r2["agree"] / r2["n"],
            "acc_fused": r2["correct30"] / r2["n"],
            "acc_unfused": r2["unfused30"] / r2["n"],
        }
        print(f"  agreement {out['e5']['pred_agreement']*100:.2f}%, acc fused {out['e5']['acc_fused']*100:.2f}%"
              f" vs unfused {out['e5']['acc_unfused']*100:.2f}%")

    with open(os.path.join(HERE, "eval_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/eval_results.json")

if __name__ == "__main__":
    main()
