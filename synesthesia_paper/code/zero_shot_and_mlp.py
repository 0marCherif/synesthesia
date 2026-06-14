"""Zero-shot classification with Gemma-4 vs trained synesthetic-feature classifiers.

Zero-shot: for each text, score each candidate label by the model's mean per-token
log-probability of the label string (teacher forcing), pick argmax. Deterministic.

Trained: 5-fold CV accuracy of LogisticRegression and a small MLP on the synesthetic
features (SYN_LOW = 20, SYN_ALL = 40).

Outputs results/comparison.json (also pulls LR numbers from results.json for cross-check).
"""
import os, sys, json
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(__file__))
from questions import as_records  # noqa

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

MODEL_ID = "google/gemma-4-E4B-it"
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA, FEAT, RES = (os.path.join(ROOT, d) for d in ["data", "features", "results"])

DATASETS = ["imdb", "agnews", "bbc", "emotion"]
QRECS = as_records()
LOW = [q["qid"] for q in QRECS if q["set"] == "SYN_LOW"]
SYN = [q["qid"] for q in QRECS if q["set"] in ("SYN_LOW", "SYN_HIGH")]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# per-dataset zero-shot instruction + display names for each label_text
ZS = {
    "imdb": dict(instr="Is the sentiment of this movie review positive or negative?",
                 disp={"neg": "negative", "pos": "positive"}),
    "agnews": dict(instr="Which category best describes this news text? "
                         "Choose one of: World, Sports, Business, Sci/Tech.",
                   disp={"World": "World", "Sports": "Sports", "Business": "Business", "Sci/Tech": "Sci/Tech"}),
    "bbc": dict(instr="Which category best describes this news article? "
                      "Choose one of: business, entertainment, politics, sport, tech.",
                disp={k: k for k in ["business", "entertainment", "politics", "sport", "tech"]}),
    "emotion": dict(instr="Which emotion does this text express? "
                          "Choose one of: sadness, joy, love, anger, fear, surprise.",
                    disp={k: k for k in ["sadness", "joy", "love", "anger", "fear", "surprise"]}),
}


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


LETTERS = ["A", "B", "C", "D", "E", "F"]


def letter_ids(tok):
    ids = {}
    for L in LETTERS:
        s = set()
        for v in (L, " " + L):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1:
                s.add(e[0])
        ids[L] = sorted(s)
    return ids


@torch.no_grad()
def zero_shot(tok, m):
    """Closed-set multiple-choice zero-shot: read the logit of each option letter (one forward/text)."""
    lid = letter_ids(tok)
    out = {}
    for name in DATASETS:
        df = pd.read_parquet(os.path.join(DATA, f"{name}.parquet")).reset_index(drop=True)
        disp = ZS[name]["disp"]; instr = ZS[name]["instr"]
        cands = list(dict.fromkeys(disp.values()))  # preserve order, unique
        letters = LETTERS[:len(cands)]
        opt_block = "\n".join(f"({L}) {c}" for L, c in zip(letters, cands))
        lt = {L: torch.tensor(lid[L], device=m.device) for L in letters}
        correct = 0; pred_counts = {L: 0 for L in letters}
        for ri in range(len(df)):
            text = df.at[ri, "text"]
            gold_c = disp[df.at[ri, "label_text"]]
            gold_L = letters[cands.index(gold_c)]
            msgs = [{"role": "user", "content":
                     f"{instr}\n\nText: {text}\n\nOptions:\n{opt_block}\n\n"
                     f"Answer with the letter of the correct option only."}]
            ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(ptext, return_tensors="pt", add_special_tokens=False).to(m.device)
            logits = m(**enc).logits[0, -1].float()
            scores = {L: torch.logsumexp(logits[lt[L]], 0).item() for L in letters}
            pred = max(scores, key=scores.get)
            pred_counts[pred] += 1
            correct += int(pred == gold_L)
        acc = correct / len(df)
        out[name] = dict(zero_shot_acc=acc, n=len(df), pred_dist=pred_counts)
        print(f"  zero-shot {name}: {acc:.3f}  pred_dist={pred_counts}")
    return out


def trained(name):
    fdf = pd.read_parquet(os.path.join(FEAT, f"{name}_features.parquet"))
    fdf[SYN] = fdf[SYN].astype("float64")
    y = np.asarray(fdf["label"], dtype=np.int64)
    res = {}
    for tag, cols in [("low", LOW), ("all", SYN)]:
        X = fdf[cols].to_numpy(dtype=np.float64)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000,
                                                            random_state=0))
        res[f"{tag}_lr"] = float(cross_val_score(lr, X, y, cv=CV, scoring="accuracy").mean())
        res[f"{tag}_mlp"] = float(cross_val_score(mlp, X, y, cv=CV, scoring="accuracy").mean())
    return res


def main():
    tok, m = load()
    zs = zero_shot(tok, m)
    comp = {}
    chance = {}
    meta = json.load(open(os.path.join(DATA, "dataset_meta.json")))
    for name in DATASETS:
        df = pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))
        chance[name] = float(df["label_text"].value_counts(normalize=True).max())
        tr = trained(name)
        comp[name] = dict(chance=chance[name], zero_shot=zs[name]["zero_shot_acc"], **tr)
        print(f"{name}: {comp[name]}")
    json.dump(comp, open(os.path.join(RES, "comparison.json"), "w"), indent=2)
    print("\nsaved results/comparison.json")
    # pretty table
    print(f"\n{'dataset':9s} {'chance':>7s} {'zeroshot':>9s} {'low_lr':>7s} {'low_mlp':>8s} {'all_lr':>7s} {'all_mlp':>8s}")
    for name in DATASETS:
        c = comp[name]
        print(f"{name:9s} {c['chance']:7.3f} {c['zero_shot']:9.3f} {c['low_lr']:7.3f} "
              f"{c['low_mlp']:8.3f} {c['all_lr']:7.3f} {c['all_mlp']:8.3f}")


if __name__ == "__main__":
    main()
