#!/usr/bin/env python3
"""
Tokenizer-parity measurement for English-Tamil (Section VI-E of the paper).

DATA (FLORES-200 devtest, 1,012 parallel sentences):
  Download the official archive (check the FLORES repo README for the current link):
    https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
  then use:
    flores200_dataset/devtest/eng_Latn.devtest
    flores200_dataset/devtest/tam_Taml.devtest

INSTALL:
  pip install tiktoken transformers sentencepiece protobuf numpy
  (Llama 3 and Gemma 2 tokenizers are gated on Hugging Face: accept their licences and run `huggingface-cli login`.
   Any tokenizer that fails to load is skipped and reported; delete its row from Table III.)

RUN:
  python tokenizer_parity.py --en eng_Latn.devtest --ta tam_Taml.devtest --out results
  python tokenizer_parity.py --selftest          # logic check only, no downloads, numbers are meaningless

OUTPUT: results.csv, results_table_rows.txt (paste into Table III), results_meta.json (versions, for reproducibility).

DEFINITIONS (per tokenizer, special tokens excluded):
  tokens/sent      mean tokens per sentence
  Tamil fertility  total Tamil tokens / total whitespace-delimited Tamil words
  parity           mean over sentences of tokens(TA)/tokens(EN), with a 95% bootstrap CI over sentences
  chars/token      total Tamil characters / total Tamil tokens
"""
import argparse, csv, json, sys, platform
import numpy as np

HF_TOKENIZERS = [
    ("Llama 3",      "meta-llama/Meta-Llama-3-8B"),
    ("Mistral 7B",   "mistralai/Mistral-7B-v0.1"),
    ("Qwen2.5",      "Qwen/Qwen2.5-7B"),
    ("Gemma 2",      "google/gemma-2-9b"),
    ("XLM-R",        "FacebookAI/xlm-roberta-base"),
    ("mT5",          "google/mt5-base"),
    ("MuRIL",        "google/muril-base-cased"),
    ("IndicBERT v2", "ai4bharat/IndicBERTv2-MLM-only"),
    ("Tamil-Llama",  "abhinand/tamil-llama-7b-base-v0.1"),
]
TIKTOKEN = [("GPT-4o (o200k_base)", "o200k_base"), ("GPT-4 (cl100k_base)", "cl100k_base")]

def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]

def load_tokenizers(selftest=False):
    """Return list of (label, encode_fn, vocab_size). encode_fn(text)->list of ids/tokens, no special tokens."""
    out = []
    if selftest:
        out.append(("DUMMY-whitespace", lambda s: s.split(), 0))
        out.append(("DUMMY-char", lambda s: list(s), 0))
        return out
    try:
        import tiktoken
        for label, name in TIKTOKEN:
            try:
                enc = tiktoken.get_encoding(name)
                out.append((label, lambda s, e=enc: e.encode(s, disallowed_special=()), enc.n_vocab))
            except Exception as e:
                print(f"[skip] {label}: {e}", file=sys.stderr)
    except ImportError:
        print("[skip] tiktoken not installed", file=sys.stderr)
    try:
        from transformers import AutoTokenizer
        for label, mid in HF_TOKENIZERS:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                out.append((label, lambda s, t=tok: t.encode(s, add_special_tokens=False), len(tok)))
            except Exception as e:
                print(f"[skip] {label} ({mid}): {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
    except ImportError:
        print("[skip] transformers not installed", file=sys.stderr)
    return out

def measure(encode, en, ta, n_boot=1000, seed=0):
    ne = np.array([max(len(encode(s)), 1) for s in en], dtype=float)
    nt = np.array([max(len(encode(s)), 1) for s in ta], dtype=float)
    words_ta = np.array([max(len(s.split()), 1) for s in ta], dtype=float)
    chars_ta = np.array([len(s) for s in ta], dtype=float)
    ratio = nt / ne
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratio), size=(n_boot, len(ratio)))
    boots = ratio[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(tok_en=ne.mean(), tok_ta=nt.mean(), fertility_ta=nt.sum()/words_ta.sum(),
                parity=ratio.mean(), parity_lo=lo, parity_hi=hi, chars_per_tok_ta=chars_ta.sum()/nt.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en"); ap.add_argument("--ta"); ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        en = ["The cat sat on the mat.", "Language models use tokens.", "Hello world"] * 20
        ta = ["பூனை பாயில் அமர்ந்தது.", "மொழி மாதிரிகள் டோக்கன்களைப் பயன்படுத்துகின்றன.", "வணக்கம் உலகம்"] * 20
    else:
        if not (a.en and a.ta): ap.error("--en and --ta are required (or use --selftest)")
        en, ta = read_lines(a.en), read_lines(a.ta)
    assert len(en) == len(ta), f"line count mismatch: {len(en)} vs {len(ta)}"
    if not a.selftest and len(en) != 1012:
        print(f"[warn] expected 1012 devtest sentences, got {len(en)}", file=sys.stderr)
    rows = []
    for label, enc, vocab in load_tokenizers(a.selftest):
        m = measure(enc, en, ta); m.update(tokenizer=label, vocab=vocab); rows.append(m)
        print(f"{label:22s} parity={m['parity']:.2f} [{m['parity_lo']:.2f},{m['parity_hi']:.2f}] fertility_TA={m['fertility_ta']:.2f}")
    if not rows: sys.exit("No tokenizer could be loaded.")
    cols = ["tokenizer","vocab","tok_en","tok_ta","fertility_ta","parity","parity_lo","parity_hi","chars_per_tok_ta"]
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r[c] for c in cols}) for r in rows]
    with open(a.out + "_table_rows.txt", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['tokenizer']} | {r['vocab']:,} | {r['tok_en']:.1f} | {r['tok_ta']:.1f} | {r['fertility_ta']:.2f} | "
                    f"{r['parity']:.2f} [{r['parity_lo']:.2f}, {r['parity_hi']:.2f}] | {r['chars_per_tok_ta']:.2f}\n")
    meta = dict(python=platform.python_version(), numpy=np.__version__, n_sentences=len(en), selftest=a.selftest)
    for mod in ("tiktoken", "transformers", "tokenizers", "sentencepiece"):
        try: meta[mod] = __import__(mod).__version__
        except Exception: pass
    json.dump(meta, open(a.out + "_meta.json", "w"), indent=1)
    print("wrote", a.out + ".csv,", a.out + "_table_rows.txt,", a.out + "_meta.json")

if __name__ == "__main__":
    main()
