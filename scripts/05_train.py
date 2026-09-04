"""Train GPT-2 Small on a corpus, saving checkpoints at BabyLM word-exposure
intervals plus one at initialisation.

The corpus is presented in file order. Word budgets are counted in words,
converted from tokens using the corpus's measured tokens-per-word ratio."""

import argparse, json, math, os, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
                          get_cosine_schedule_with_warmup)

CHECKPOINT_SCHEDULE = (list(range(1_000_000, 10_000_001, 1_000_000)) +
                       list(range(20_000_000, 100_000_001, 10_000_000)))

def build_cache(corpus, cache_path, tok_path, block=512):
    from array import array
    import numpy as np
    tok = GPT2TokenizerFast.from_pretrained(tok_path)
    flat = array('i'); total_words = total_tokens = 0
    with open(corpus) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            ids = tok.encode(s) + [tok.eos_token_id]
            total_words += len(s.split()); total_tokens += len(ids)
            flat.extend(ids)
    n_blocks = len(flat) // block
    arr = np.frombuffer(flat, dtype=np.int32)[:n_blocks * block].reshape(n_blocks, block)
    t = torch.from_numpy(arr.copy())
    blocks = arr
    meta = {"n_blocks": len(blocks), "total_words": total_words,
            "total_tokens": total_tokens, "words_per_token": total_words/total_tokens}
    torch.save({"blocks": t, "meta": meta}, cache_path)
    print(f"cache -> {cache_path}: {meta}", flush=True)
    return t, meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--cache_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--max_words", type=int, default=9_000_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_tokens", type=int, default=16_384)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--build_cache_only", action="store_true")
    p.add_argument("--tokenizer_path", default="gpt2",
                   help="local dir; compute nodes have no internet")
    a = p.parse_args()

    print(f"ARGS: {vars(a)}", flush=True)               # gotcha #7: echo args
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    assert os.access(out, os.W_OK), f"not writable: {out}"

    if Path(a.cache_path).exists():
        d = torch.load(a.cache_path); blocks, meta = d["blocks"], d["meta"]
        print(f"cache hit: {meta}", flush=True)
    else:
        blocks, meta = build_cache(a.corpus, a.cache_path, a.tokenizer_path)
    if a.build_cache_only:
        print("cache built; exiting."); return

    wpt = meta["words_per_token"]
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "no GPU visible"

    cfg = GPT2Config(vocab_size=GPT2TokenizerFast.from_pretrained(a.tokenizer_path).vocab_size,
                     n_positions=512, n_ctx=512, n_embd=768, n_layer=12, n_head=12)
    model = GPT2LMHeadModel(cfg).to(dev)

    bs = max(1, a.batch_tokens // 512)
    ds = TensorDataset(blocks.long())
    loader = DataLoader(ds, batch_size=bs, shuffle=False, pin_memory=True)

    words_per_step = bs * 512 * wpt
    max_steps = math.ceil(min(a.max_words, meta["total_words"]*a.epochs) / words_per_step)
    print(f"words/token={wpt:.4f}  words/step={words_per_step:.0f}  "
          f"steps/epoch={len(loader)}  max_steps={max_steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, max(1, max_steps//100), max_steps)

    words_seen = step = 0; nxt = 0; log = []
    tok = GPT2TokenizerFast.from_pretrained(a.tokenizer_path); tok.pad_token = tok.eos_token

    # checkpoint, by which time a curriculum has already taught its front-loaded
    # words -- their surprisal has no room to fall 50% further, so they never cross
    # the threshold, get censored to the MAXIMUM, and are recorded as "learned last"
    # when they were learned first. An untrained reference makes s0 uniformly high
    # for every word, so the 50%-of-range criterion is well defined regardless of
    # when in training a word is taught.
    d0 = out / "checkpoint_0M"
    model.save_pretrained(d0); tok.save_pretrained(d0)
    print("  ckpt 0M (init reference)", flush=True)

    model.train(); t0 = time.time()
    for ep in range(a.epochs):
        for (x,) in loader:
            if words_seen >= a.max_words or step >= max_steps: break
            x = x.to(dev, non_blocking=True)
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            words_seen += words_per_step; step += 1
            while nxt < len(CHECKPOINT_SCHEDULE) and words_seen >= CHECKPOINT_SCHEDULE[nxt]:
                cw = CHECKPOINT_SCHEDULE[nxt]
                d = out / f"checkpoint_{cw//1_000_000}M"
                model.save_pretrained(d); tok.save_pretrained(d)
                log.append({"words": cw, "step": step, "loss": round(loss.item(),4),
                            "elapsed_s": round(time.time()-t0)})
                print(f"  ckpt {cw//1_000_000}M | step {step} | loss {loss.item():.4f} "
                      f"| {(time.time()-t0)/60:.1f}min", flush=True)
                json.dump({"run": a.run_name, "seed": a.seed, "meta": meta,
                           "checkpoints": log}, open(out/"training_log.json","w"), indent=2)
                nxt += 1
        if words_seen >= a.max_words or step >= max_steps: break

    model.save_pretrained(out/"final"); tok.save_pretrained(out/"final")
    # gotcha #7: postconditions
    assert (out/"final"/"model.safetensors").stat().st_size > 1e8, "final model too small"
    assert (out/"checkpoint_0M").is_dir(), "missing init checkpoint"
    assert len(log) >= 10, f"expected >=10 checkpoints, got {len(log)}"
    print(f"DONE steps={step} words_seen={words_seen:,.0f} ckpts={len(log)}", flush=True)

if __name__ == "__main__":
    main()
