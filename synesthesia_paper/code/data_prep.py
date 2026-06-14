"""Prepare balanced, truncated subsets of 4 classification datasets.

Output: data/<name>.parquet with columns [text, label, label_text].
Also writes data/dataset_meta.json with class names and sizes.
"""
import json, re, os
import pandas as pd
from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

N_TOTAL = 500
MAX_WORDS = 300
SEED = 12345

DATASETS = {
    "imdb":    dict(path="stanfordnlp/imdb", config="plain_text", split="train"),
    "agnews":  dict(path="fancyzhx/ag_news", config="default",    split="train"),
    "bbc":     dict(path="SetFit/bbc-news",  config="default",    split="train"),
    "emotion": dict(path="dair-ai/emotion",  config="split",      split="train"),
}


def truncate(text, max_words=MAX_WORDS):
    text = re.sub(r"<br\s*/?>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


def balanced_subset(df, n_total, seed=SEED):
    labels = sorted(df["label"].unique())
    per = n_total // len(labels)
    parts = []
    for lab in labels:
        sub = df[df["label"] == lab]
        take = min(per, len(sub))
        parts.append(sub.sample(n=take, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    meta = {}
    for name, cfg in DATASETS.items():
        print(f"\n=== {name} ===")
        ds = load_dataset(cfg["path"], cfg["config"], split=cfg["split"])
        names = None
        try:
            names = ds.features["label"].names
        except Exception:
            pass
        df = ds.to_pandas()
        # ensure text col
        if "text" not in df.columns:
            raise ValueError(f"{name}: no text column, has {df.columns.tolist()}")
        df = df[["text", "label"] + (["label_text"] if "label_text" in df.columns else [])].copy()
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].map(lambda t: truncate(t))
        df = df[df["text"].str.len() > 0]
        sub = balanced_subset(df, N_TOTAL)
        if names is not None and "label_text" not in sub.columns:
            sub["label_text"] = sub["label"].map(lambda i: names[int(i)])
        elif "label_text" not in sub.columns:
            sub["label_text"] = sub["label"].astype(str)
        if names is None:
            names = sorted(sub["label_text"].unique().tolist())
        path = os.path.join(OUT, f"{name}.parquet")
        sub.to_parquet(path, index=False)
        dist = sub["label_text"].value_counts().to_dict()
        meta[name] = dict(path=os.path.relpath(path), n=len(sub), classes=names,
                          n_classes=len(names), dist=dist,
                          mean_words=float(sub["text"].str.split().map(len).mean()))
        print(f"  n={len(sub)} classes={names}")
        print(f"  dist={dist}")
        print(f"  mean words={meta[name]['mean_words']:.1f}")
    with open(os.path.join(OUT, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nWrote", os.path.join(OUT, "dataset_meta.json"))


if __name__ == "__main__":
    main()
