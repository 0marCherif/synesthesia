"""Independent metaphoricity measure for each sensory predicate.

We ask the model how often the adjective is used FIGURATIVELY (non-physically), via a
yes/no probe, and read logit(yes)-logit(no). This gives a model-based metaphoricity score
that is independent of the authors' a-priori judgments. Saved to results/metaphoricity_llm.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(__file__))
from questions import SYN_LOW, SYN_HIGH  # noqa

MODEL_ID = "google/gemma-4-E4B-it"
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def load():
    import transformers
    from transformers import AutoProcessor, AutoTokenizer
    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        tok = getattr(proc, "tokenizer", None) or AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    return tok, m


def ids_for(tok, words):
    s = set()
    for w in words:
        for v in (w, " " + w):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1:
                s.add(e[0])
    return sorted(s)


def main():
    tok, m = load()
    yes = torch.tensor(ids_for(tok, ["yes", "Yes", "YES"]), device=m.device)
    no = torch.tensor(ids_for(tok, ["no", "No", "NO"]), device=m.device)
    rows = []
    preds = [(s, qid, pred, "SYN_LOW", meta) for (s, qid, pred, q, meta) in SYN_LOW] + \
            [(s, qid, pred, "SYN_HIGH", meta) for (s, qid, pred, q, meta) in SYN_HIGH]
    for sense, qid, pred, st, apriori in preds:
        prompt = (f"When the adjective \"{pred}\" is used in everyday English, is it often used "
                  f"figuratively or metaphorically to describe something non-physical "
                  f"(for example a mood, a person, or a situation)?")
        msgs = [{"role": "user", "content": prompt + "\nAnswer with only 'yes' or 'no'."}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(m.device)
        with torch.no_grad():
            lg = m(**enc).logits[0, -1].float()
        score = (torch.logsumexp(lg[yes], 0) - torch.logsumexp(lg[no], 0)).item()
        rows.append(dict(qid=qid, sense=sense, predicate=pred, set=st,
                         a_priori_metaphoricity=apriori, llm_metaphoricity_logit=score))
        print(f"{pred:10s} {st:9s} apriori={apriori:.2f} llm_logit={score:+.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metaphoricity_llm.csv"), index=False)
    print("\nsaved results/metaphoricity_llm.csv")


if __name__ == "__main__":
    main()
