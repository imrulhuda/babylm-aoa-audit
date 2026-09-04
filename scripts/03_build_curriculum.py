"""Build the AoA curriculum: bin scored sentences into five developmental
stages, shuffle within each stage, and concatenate in stage order with the
earlier stages repeated."""

import argparse, json, random
from collections import defaultdict

STAGE_NAMES = ["stage1_le18", "stage2_19_21", "stage3_22_24", "stage4_25_27", "stage5_ge28"]
STAGE_REPEATS = [3, 2, 1, 1, 1]

def stage_of(a):
    if a <= 18: return 0
    if a <= 21: return 1
    if a <= 24: return 2
    if a <= 27: return 3
    return 4

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_path",  required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--stats_out", required=True)
    p.add_argument("--within_stage_seed", type=int, default=20260903)
    a = p.parse_args()

    stages = defaultdict(list)
    total = 0
    with open(a.in_path) as f:
        for line in f:
            o = json.loads(line)
            stages[stage_of(o["median_aoa"])].append(o["text"])
            total += 1

    binned = sum(len(v) for v in stages.values())
    assert binned == total, f"BIN GAP: {total-binned} sentences unassigned"
    print(f"scored={total:,}  binned={binned:,}  dropped=0  [F fixed]")

    rng = random.Random(a.within_stage_seed)
    for i in range(5):
        rng.shuffle(stages[i])
    print(f"within-stage shuffle seed={a.within_stage_seed}  [G fixed]")

    stats, written = {}, 0
    with open(a.out_path, "w") as out:
        for i, name in enumerate(STAGE_NAMES):
            n = len(stages[i]); w = sum(len(t.split()) for t in stages[i])
            stats[name] = {"sentences": n, "words": w, "repeats": STAGE_REPEATS[i]}
            print(f"  {name}: {n:>8,} sentences  {w:>10,} words  x{STAGE_REPEATS[i]}")
            for _ in range(STAGE_REPEATS[i]):
                for t in stages[i]:
                    out.write(t + "\n"); written += 1
    stats["total_written"] = written
    stats["within_stage_seed"] = a.within_stage_seed
    json.dump(stats, open(a.stats_out, "w"), indent=2)
    print(f"wrote {written:,} lines -> {a.out_path}")

if __name__ == "__main__":
    main()
