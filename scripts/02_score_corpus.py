"""Score every sentence in a BabyLM corpus by the mean AoA of the CDI words it
contains. Sentences with no CDI word are assigned an AoA of 31 months.

CHILDES speaker-code prefixes are stripped before tokenising so that codes
such as *ANT: are not read as ordinary words."""

import argparse, json, re
from pathlib import Path

SOURCES = {
    "childes": "childes.train.txt", "bnc_spoken": "bnc_spoken.train.txt",
    "switchboard": "switchboard.train.txt", "gutenberg": "gutenberg.train.txt",
    "open_subtitles": "open_subtitles.train.txt", "simple_wiki": "simple_wiki.train.txt",
}
tok = re.compile(r"[a-z']+")
SPEAKER = re.compile(r"^\*[A-Za-z]{2,4}:\s*")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--aoa_norms", required=True)
    p.add_argument("--out_path", required=True)
    a = p.parse_args()
    aoa = json.load(open(a.aoa_norms))

    def sentence_aoa(text):
        words = tok.findall(text.lower())
        scored = [aoa[w] for w in words if w in aoa]
        if not scored:
            return 31.0, 0, len(words)
        return sum(scored) / len(scored), len(scored), len(words)

    total = 0
    with open(a.out_path, "w") as out:
        for source, fname in SOURCES.items():
            path = Path(a.data_dir) / fname
            print(f"scoring {source} ...", flush=True)
            with open(path) as f:
                for line in f:
                    text = SPEAKER.sub("", line.strip())
                    if not text:
                        continue
                    med, n_known, n_words = sentence_aoa(text)
                    out.write(json.dumps({"text": text, "source": source,
                                          "median_aoa": round(med, 2),
                                          "n_known": n_known, "n_words": n_words}) + "\n")
                    total += 1
    print(f"scored {total:,} sentences -> {a.out_path}")

if __name__ == "__main__":
    main()
