"""Estimate each CDI word's model age of acquisition from its surprisal
trajectory across checkpoints, and correlate it with human AoA."""

import argparse, json, os, re, random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from scipy import stats

WORD_RE = re.compile(r"[a-z']+")


def find_checkpoints(ckpt_dir):
    ckpts = []
    for d in sorted(Path(ckpt_dir).iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"checkpoint_(\d+)M", d.name)
        if m:
            ckpts.append((int(m.group(1)), d))
    ckpts.sort(key=lambda x: x[0])
    return ckpts  # list of (words_in_M, path), final/ excluded on purpose (~duplicate of last checkpoint)


def collect_word_contexts(scored_path, aoa_words, max_examples, min_examples, seed):
    """word -> list of raw sentence texts containing that word as a whole token."""
    pools = defaultdict(list)
    with open(scored_path) as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            seen_in_line = set()
            for m in WORD_RE.finditer(text.lower()):
                w = m.group(0)
                if w in aoa_words and w not in seen_in_line:
                    pools[w].append(text)
                    seen_in_line.add(w)

    rng = random.Random(seed)
    contexts, freq, excluded = {}, {}, []
    for w in aoa_words:
        examples = pools.get(w, [])
        freq[w] = len(examples)
        if len(examples) < min_examples:
            excluded.append(w)
            continue
        rng.shuffle(examples)
        contexts[w] = examples[:max_examples]
    return contexts, freq, excluded


def word_span(text, word):
    """Character (start, end) of the first whole-token occurrence of `word` in text (case-insensitive)."""
    for m in WORD_RE.finditer(text.lower()):
        if m.group(0) == word:
            return m.start(), m.end()
    return None


@torch.no_grad()
def surprisal_for_word(model, tokenizer, device, text, word):
    span = word_span(text, word)
    if span is None:
        return None
    start, end = span

    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return None

    # token indices whose span overlaps the word's character span
    tok_idxs = [i for i, (s, e) in enumerate(offsets) if e > start and s < end and not (s == 0 and e == 0)]
    if not tok_idxs or tok_idxs[0] == 0:
        return None  # word's first token has no left context to condition on

    logits = model(input_ids=input_ids).logits[0]  # [T, V]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    surprisal_bits = 0.0
    for t in tok_idxs:
        target_id = input_ids[0, t]
        lp = log_probs[t - 1, target_id].item()
        surprisal_bits += -lp / 0.6931471805599453  # nats -> bits
    return surprisal_bits


def model_aoa_from_curve(checkpoints_m, surprisals):
    """First checkpoint where surprisal drops to <=50% of the range from the
    first checkpoint's value down to the curve's minimum. Censored (None) if never crossed."""
    s0 = surprisals[0]
    s_min = min(surprisals)
    if s0 == s_min:
        return None
    threshold = s0 - 0.5 * (s0 - s_min)
    for m, s in zip(checkpoints_m, surprisals):
        if s <= threshold:
            return m
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", default=str(Path(__file__).parent.parent / "checkpoints/aoa_curriculum"))
    p.add_argument("--aoa_norms", default=str(Path(__file__).parent.parent / "data/aoa_norms.json"))
    p.add_argument("--scored_sentences", default=str(Path(__file__).parent.parent / "data/scored_sentences.jsonl"))
    p.add_argument("--out_dir", default=str(Path(__file__).parent.parent / "results"))
    p.add_argument("--max_examples_per_word", type=int, default=20)
    p.add_argument("--min_examples_per_word", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoints", default=None,
                   help="comma-separated list of checkpoint sizes in M to score, e.g. '1,5,9' (default: all found)")
    p.add_argument("--limit_words", type=int, default=None,
                   help="only analyze the first N CDI words (for smoke-testing)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    aoa_norms = json.load(open(args.aoa_norms))
    if args.limit_words:
        aoa_norms = dict(list(aoa_norms.items())[:args.limit_words])
    checkpoints = find_checkpoints(args.checkpoints_dir)
    if args.checkpoints:
        wanted = {int(x) for x in args.checkpoints.split(",")}
        checkpoints = [(m, p_) for m, p_ in checkpoints if m in wanted]
    if not checkpoints:
        raise SystemExit(f"No matching checkpoint_*M directories found in {args.checkpoints_dir}")
    checkpoints_m = [m for m, _ in checkpoints]
    print(f"Found {len(checkpoints)} checkpoints: {checkpoints_m}M")

    print("Collecting contexts per CDI word ...")
    contexts, freq, excluded = collect_word_contexts(
        args.scored_sentences, set(aoa_norms), args.max_examples_per_word,
        args.min_examples_per_word, args.seed,
    )
    print(f"  {len(contexts)}/{len(aoa_norms)} words have >= {args.min_examples_per_word} corpus examples")
    print(f"  {len(excluded)} words excluded for too few examples (e.g. {excluded[:10]})")

    # surprisal[word][checkpoint_m] = mean surprisal across sampled sentences
    surprisal_curves = {w: {} for w in contexts}

    for m, ckpt_path in checkpoints:
        print(f"Scoring checkpoint {m}M ({ckpt_path}) ...")
        tokenizer = GPT2TokenizerFast.from_pretrained(ckpt_path)
        model = GPT2LMHeadModel.from_pretrained(ckpt_path).to(device).eval()

        for w, sentences in contexts.items():
            vals = []
            for text in sentences:
                s = surprisal_for_word(model, tokenizer, device, text, w)
                if s is not None:
                    vals.append(s)
            if vals:
                surprisal_curves[w][m] = sum(vals) / len(vals)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # keep only words with a surprisal value at every checkpoint
    complete_words = [w for w in contexts if len(surprisal_curves[w]) == len(checkpoints_m)]
    print(f"{len(complete_words)}/{len(contexts)} words have a complete surprisal curve across all checkpoints")

    per_word = {}
    for w in complete_words:
        curve = [surprisal_curves[w][m] for m in checkpoints_m]
        m_aoa = model_aoa_from_curve(checkpoints_m, curve)
        per_word[w] = {
            "human_aoa_months": aoa_norms[w],
            "n_examples": len(contexts[w]),
            "corpus_freq": freq[w],
            "surprisal_by_checkpoint": dict(zip([f"{m}M" for m in checkpoints_m], curve)),
            "model_aoa_M_words": m_aoa,
        }

    # censor un-acquired words at one checkpoint past the last, mirroring CDI's AoA=31 censoring
    step = (checkpoints_m[1] - checkpoints_m[0]) if len(checkpoints_m) > 1 else 1
    censor_value = checkpoints_m[-1] + step
    human_aoa, model_aoa, log_freq = [], [], []
    n_censored = 0
    for w, rec in per_word.items():
        m_aoa = rec["model_aoa_M_words"]
        if m_aoa is None:
            m_aoa = censor_value
            n_censored += 1
        human_aoa.append(rec["human_aoa_months"])
        model_aoa.append(m_aoa)
        log_freq.append(torch.log(torch.tensor(float(rec["corpus_freq"] + 1))).item())

    rho, p_val = stats.spearmanr(human_aoa, model_aoa)

    # partial Spearman correlation controlling for log corpus frequency:
    # correlate the residuals of each rank-transformed variable against log-freq
    def residualize(y, x):
        y_r = stats.rankdata(y)
        x_r = stats.rankdata(x)
        slope, intercept, *_ = stats.linregress(x_r, y_r)
        return y_r - (slope * x_r + intercept)

    resid_human = residualize(human_aoa, log_freq)
    resid_model = residualize(model_aoa, log_freq)
    partial_rho, partial_p = stats.pearsonr(resid_human, resid_model)

    freq_human_rho, freq_human_p = stats.spearmanr(log_freq, human_aoa)
    freq_model_rho, freq_model_p = stats.spearmanr(log_freq, model_aoa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_words_analyzed": len(complete_words),
        "n_words_excluded_too_few_examples": len(excluded),
        "n_words_censored_never_acquired": n_censored,
        "checkpoints_M_words": checkpoints_m,
        "spearman_model_vs_human_aoa": {"rho": rho, "p": p_val},
        "spearman_model_aoa_vs_log_freq": {"rho": freq_model_rho, "p": freq_model_p},
        "spearman_human_aoa_vs_log_freq": {"rho": freq_human_rho, "p": freq_human_p},
        "partial_correlation_controlling_for_freq": {"rho": partial_rho, "p": partial_p},
        "per_word": per_word,
    }
    with open(out_dir / "aoa_trajectory.json", "w") as f:
        json.dump(results, f, indent=2)

    report = f"""AoA Trajectory Analysis
========================
Checkpoints analyzed: {checkpoints_m}M words
Words analyzed: {len(complete_words)} / {len(aoa_norms)} CDI words
  excluded (too few corpus examples): {len(excluded)}
  censored (never crossed 50% surprisal-drop threshold): {n_censored}

Spearman correlation: model AoA (M words) vs human AoA (months)
  rho = {rho:.3f}, p = {p_val:.4g}

Confound check: does frequency alone explain each AoA measure?
  log-freq vs human AoA:  rho = {freq_human_rho:.3f}, p = {freq_human_p:.4g}
  log-freq vs model AoA:  rho = {freq_model_rho:.3f}, p = {freq_model_p:.4g}

Partial correlation (model vs human AoA, controlling for log-freq):
  rho = {partial_rho:.3f}, p = {partial_p:.4g}
"""
    with open(out_dir / "aoa_trajectory_report.txt", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
