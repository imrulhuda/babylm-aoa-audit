"""Build the flat-shuffle baseline: the same multiset of lines as the
curriculum, in random order."""

import argparse, os, random
from pathlib import Path

DEFAULT_IN  = Path(__file__).parent.parent / "data/curriculum_train.txt"
DEFAULT_OUT = Path(__file__).parent.parent / "data/flat_shuffle_train.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", default=str(DEFAULT_IN))
    p.add_argument("--out_path", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=1337)  # distinct from training seed (42)
    args = p.parse_args()

    with open(args.in_path) as f:
        lines = f.readlines()

    n_lines_in = len(lines)
    n_words_in = sum(len(l.split()) for l in lines)

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        f.writelines(lines)

    print(f"Read {n_lines_in:,} lines / {n_words_in:,} words from {args.in_path}")
    print(f"Shuffled with seed={args.seed}")
    print(f"Wrote {len(lines):,} lines to {args.out_path}")
    assert len(lines) == n_lines_in, "line count must be unchanged -- same multiset, different order"


if __name__ == "__main__":
    main()
