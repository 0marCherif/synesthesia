"""Extract synesthetic features: feature(text, q) = logsumexp(yes logits) - logsumexp(no logits).

For each dataset, produces features/<name>_features.parquet:
  columns = [<qid_1> ... <qid_50>, label, label_text]
Saves checkpoints per dataset. Deterministic forward passes (no sampling).
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from questions import as_records, SENSES  # noqa

MODEL_ID = "google/gemma-4-E4B-it"
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
FEAT = os.path.join(ROOT, "features")
os.makedirs(FEAT, exist_ok=True)

DATASETS = ["agnews", "emotion", "imdb", "bbc"]  # short -> long
TOKEN_BUDGET = 24000  # max tokens (sum of padded lengths) per batch


def load_model():
    import transformers
    from transformers import AutoProcessor, AutoTokenizer
    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        tok = getattr(proc, "tokenizer", None) or AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def yes_no_ids(tok):
    def ids_for(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                enc = tok.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    s.add(enc[0])
        return sorted(s)
    return ids_for(["yes", "Yes", "YES"]), ids_for(["no", "No", "NO"])


def build_prompt(tok, text, question):
    msgs = [{"role": "user",
             "content": f"{text}\n\nQuestion: {question}\nAnswer with only 'yes' or 'no'."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    t_start = time.time()
    qrecs = as_records()
    qids = [q["qid"] for q in qrecs]
    print(f"{len(qids)} questions")

    tok, model = load_model()
    yes_ids, no_ids = yes_no_ids(tok)
    yes_t = torch.tensor(yes_ids, device=model.device)
    no_t = torch.tensor(no_ids, device=model.device)
    print("loaded model; yes/no ids:", yes_ids, no_ids)

    for name in DATASETS:
        out_path = os.path.join(FEAT, f"{name}_features.parquet")
        if os.path.exists(out_path):
            print(f"[skip] {name} already done")
            continue
        df = pd.read_parquet(os.path.join(DATA, f"{name}.parquet")).reset_index(drop=True)
        n = len(df)
        print(f"\n=== {name}: {n} texts x {len(qids)} questions = {n*len(qids)} prompts ===")

        # Build all prompts and tokenize once
        items = []  # (row_idx, qcol, input_ids)
        for ri in range(n):
            text = df.at[ri, "text"]
            for qi, q in enumerate(qrecs):
                ids = tok(build_prompt(tok, text, q["question"]), add_special_tokens=False).input_ids
                items.append((ri, qi, ids))
        # sort by length for efficient padding
        items.sort(key=lambda x: len(x[2]))
        feat = np.full((n, len(qids)), np.nan, dtype=np.float32)

        # dynamic batching by token budget
        t0 = time.time()
        i = 0
        done = 0
        while i < len(items):
            j = i
            maxlen = 0
            while j < len(items):
                cand = max(maxlen, len(items[j][2]))
                bs = j - i + 1
                if bs * cand > TOKEN_BUDGET and bs > 1:
                    break
                maxlen = cand
                j += 1
            batch = items[i:j]
            pad_id = tok.pad_token_id
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for b, (_, _, ids) in enumerate(batch):
                L = len(ids)
                input_ids[b, maxlen - L:] = torch.tensor(ids, dtype=torch.long)
                attn[b, maxlen - L:] = 1
            input_ids = input_ids.to(model.device); attn = attn.to(model.device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :].float()
            ly = torch.logsumexp(logits[:, yes_t], dim=1)
            ln = torch.logsumexp(logits[:, no_t], dim=1)
            diff = (ly - ln).cpu().numpy()
            for b, (ri, qi, _) in enumerate(batch):
                feat[ri, qi] = diff[b]
            done += len(batch)
            i = j
            if done % 5000 < len(batch):
                el = time.time() - t0
                print(f"  {done}/{len(items)}  {done/el:.0f} prompts/s  elapsed {el:.0f}s", flush=True)

        fdf = pd.DataFrame(feat, columns=qids)
        fdf["label"] = df["label"].values
        fdf["label_text"] = df["label_text"].values
        fdf.to_parquet(out_path, index=False)
        # sanity: cross-text std per question
        stds = fdf[qids].std(axis=0)
        print(f"  saved {out_path}; per-question cross-text std: "
              f"min={stds.min():.3f} median={stds.median():.3f} max={stds.max():.3f}")
        print(f"  dataset time: {time.time()-t0:.0f}s")

    print(f"\nALL DONE in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
