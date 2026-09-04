"""Derive per-word age-of-acquisition norms from the CDI production data.

A word's AoA is the earliest age at which the proportion of children
producing it reaches 50%. Words never reaching 50% are assigned 31 months.
"""

import argparse, csv, json
from collections import Counter


def build_norms(cdi_path):
    """Return {word: AoA in months} from a CDI production CSV."""
    norms = {}
    with open(cdi_path) as f:
        reader = csv.DictReader(f)
        age_cols = [c for c in reader.fieldnames if c.isdigit()]
        for row in reader:
            word = row["word"].strip().lower()
            aoa = 31
            for age in age_cols:
                prop = float(row[age]) if row[age] else 0.0
                if prop >= 0.5:
                    aoa = int(age)
                    break
            norms[word] = aoa
    return norms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cdi_csv", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    norms = build_norms(a.cdi_csv)
    json.dump(norms, open(a.out, "w"), indent=2)
    print(f"{len(norms):,} words")
    print(f"AoA distribution: {dict(sorted(Counter(norms.values()).items()))}")


if __name__ == "__main__":
    main()
