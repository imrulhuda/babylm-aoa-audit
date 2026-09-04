"""Aggregate trajectory results across seeds and compare conditions.

Reports per-seed correlations, correlations on the seed-averaged model AoA,
a paired bootstrap over words, and a run-level test across seeds."""

import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

def load(p):
    return json.load(open(p))["per_word"]

def spear(a, b):
    r, p = stats.spearmanr(a, b)
    return float(r), float(p)

def partial_spear(a, b, ctrl):
    ra, rb, rc = (stats.rankdata(x) for x in (a, b, ctrl))
    res = lambda y, x: y - np.poly1d(np.polyfit(x, y, 1))(x)
    r, p = stats.pearsonr(res(ra, rc), res(rb, rc))
    return float(r), float(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    RD = Path(a.results_dir)
    conds = ["curriculum", "flat_shuffle"]
    per_word, per_seed_rho = {}, {}

    for c in conds:
        runs = [load(RD / f"{c}_seed{s}" / "aoa_trajectory.json") for s in a.seeds]
        common = set(runs[0])
        for r in runs[1:]:
            common &= set(r)
        common = sorted(w for w in common
                        if all(r[w].get("model_aoa_M_words") is not None for r in runs))
        per_word[c] = {
            "words": common,
            "human": np.array([runs[0][w]["human_aoa_months"] for w in common], float),
            "logfreq": np.log(np.array([runs[0][w]["corpus_freq"] for w in common], float)),
            "model_by_seed": np.array([[r[w]["model_aoa_M_words"] for w in common] for r in runs], float),
        }
        per_seed_rho[c] = [spear(per_word[c]["model_by_seed"][i], per_word[c]["human"])[0]
                           for i in range(len(a.seeds))]

    common = sorted(set(per_word["curriculum"]["words"]) & set(per_word["flat_shuffle"]["words"]))
    idx = {c: [per_word[c]["words"].index(w) for w in common] for c in conds}
    human = per_word["curriculum"]["human"][idx["curriculum"]]
    logf = per_word["curriculum"]["logfreq"][idx["curriculum"]]
    mean_model = {c: per_word[c]["model_by_seed"][:, idx[c]].mean(axis=0) for c in conds}

    out = {"n_words_common": len(common), "seeds": a.seeds, "per_seed_rho": per_seed_rho}
    print(f"n words common to all 6 runs: {len(common)}\n")
    for c in conds:
        rs = per_seed_rho[c]
        r_avg, p_avg = spear(mean_model[c], human)
        pr, pp = partial_spear(mean_model[c], human, logf)
        out[c] = {"per_seed_rho": rs, "mean_rho": float(np.mean(rs)),
                  "sd_rho": float(np.std(rs, ddof=1)),
                  "rho_seed_averaged": r_avg, "p_seed_averaged": p_avg,
                  "partial_rho": pr, "partial_p": pp}
        print(f"{c}")
        print(f"  per-seed rho      : {', '.join(f'{x:+.3f}' for x in rs)}")
        print(f"  mean +/- SD       : {np.mean(rs):+.3f} +/- {np.std(rs, ddof=1):.3f}")
        print(f"  rho (seed-avg AoA): {r_avg:+.3f}  p={p_avg:.4g}")
        print(f"  partial | logfreq : {pr:+.3f}  p={pp:.4g}\n")

    obs = spear(mean_model["curriculum"], human)[0] - spear(mean_model["flat_shuffle"], human)[0]
    rng = np.random.default_rng(a.boot_seed)
    n = len(common); diffs = np.empty(a.n_bootstrap)
    for i in range(a.n_bootstrap):
        s = rng.integers(0, n, n)
        diffs[i] = (spear(mean_model["curriculum"][s], human[s])[0]
                    - spear(mean_model["flat_shuffle"][s], human[s])[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    t, p_t = stats.ttest_ind(per_seed_rho["curriculum"], per_seed_rho["flat_shuffle"])

    out["comparison"] = {"observed_diff": float(obs), "ci95": [float(lo), float(hi)],
                         "p_bootstrap": float(p_boot),
                         "run_level_t": float(t), "run_level_p": float(p_t)}
    print("curriculum - flat_shuffle (seed-averaged model AoA)")
    print(f"  observed diff : {obs:+.3f}")
    print(f"  95% CI (word bootstrap, {a.n_bootstrap}) : [{lo:+.3f}, {hi:+.3f}]  p={p_boot:.4g}")
    print(f"  run-level t-test on 3v3 per-seed rho     : t={t:.3f}  p={p_t:.4g}  (low power, correct unit)")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwritten -> {a.out}")

if __name__ == "__main__":
    main()
