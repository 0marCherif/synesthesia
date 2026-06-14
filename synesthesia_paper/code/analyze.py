"""Analyses for the synesthetic-probing short paper.

Reads features/<name>_features.parquet and data/<name>.parquet, produces:
  results/results.json        (all numbers)
  results/per_question.csv    (univariate accuracy + metaphoricity per question)
  figures/*.png               (incremental curves, metaphoricity scatter, per-sense, heatmap)
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from questions import as_records, SENSES  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA, FEAT, RES, FIG = (os.path.join(ROOT, d) for d in ["data", "features", "results", "figures"])
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)

DATASETS = ["imdb", "agnews", "bbc", "emotion"]
QRECS = as_records()
QID = [q["qid"] for q in QRECS]
QMETA = {q["qid"]: q for q in QRECS}
LOW = [q["qid"] for q in QRECS if q["set"] == "SYN_LOW"]
HIGH = [q["qid"] for q in QRECS if q["set"] == "SYN_HIGH"]
GIB = [q["qid"] for q in QRECS if q["set"] == "GIB"]
SYN = LOW + HIGH

SEED = 0
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
f1m = make_scorer(f1_score, average="macro")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def acc_cv(X, y):
    return cross_val_score(clf(), X, y, cv=CV, scoring="accuracy")


def f1_cv(X, y):
    return cross_val_score(clf(), X, y, cv=CV, scoring=f1m)


def greedy_forward(X, y, cols, max_k=None):
    """Greedy forward selection; returns list of (k, best_acc, chosen_col)."""
    max_k = max_k or len(cols)
    remaining = list(range(len(cols)))
    chosen = []
    curve = []
    while remaining and len(chosen) < max_k:
        best, best_j = -1, None
        for j in remaining:
            idx = chosen + [j]
            a = acc_cv(X[:, idx], y).mean()
            if a > best:
                best, best_j = a, j
        chosen.append(best_j)
        remaining.remove(best_j)
        curve.append((len(chosen), best, cols[best_j]))
    return curve


def main():
    results = {"datasets": {}, "config": {"seed": SEED, "n_low": len(LOW), "n_high": len(HIGH), "n_gib": len(GIB)}}
    per_q_rows = []

    for name in DATASETS:
        fp = os.path.join(FEAT, f"{name}_features.parquet")
        if not os.path.exists(fp):
            print(f"[skip] {name}: no features yet")
            continue
        fdf = pd.read_parquet(fp)
        raw = pd.read_parquet(os.path.join(DATA, f"{name}.parquet")).reset_index(drop=True)
        fdf[QID] = fdf[QID].astype("float64")
        y = np.asarray(fdf["label"], dtype=np.int64)
        nclass = len(np.unique(y))
        chance = pd.Series(y).value_counts(normalize=True).max()
        print(f"\n=== {name}: n={len(y)} classes={nclass} chance={chance:.3f} ===")

        Xall = {g: fdf[cols].to_numpy(dtype=np.float64) for g, cols in
                [("SYN_LOW", LOW), ("SYN_HIGH", HIGH), ("SYN_ALL", SYN), ("GIB", GIB)]}

        grp = {}
        for g, X in Xall.items():
            a = acc_cv(X, y); f = f1_cv(X, y)
            grp[g] = dict(acc=float(a.mean()), acc_std=float(a.std()),
                          f1=float(f.mean()), n_feat=X.shape[1])
            print(f"  {g:8s} acc={a.mean():.3f}±{a.std():.3f} f1={f.mean():.3f} ({X.shape[1]} feats)")

        # TF-IDF lexical baseline
        tfidf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True),
                              LogisticRegression(max_iter=2000))
        tf_acc = cross_val_score(tfidf, raw["text"].astype(str).tolist(), y, cv=CV, scoring="accuracy")
        grp["TFIDF"] = dict(acc=float(tf_acc.mean()), acc_std=float(tf_acc.std()), n_feat=5000)
        print(f"  {'TFIDF':8s} acc={tf_acc.mean():.3f}±{tf_acc.std():.3f}")

        # Univariate accuracy per synesthetic question
        uni = {}
        for c in SYN + GIB:
            a = acc_cv(fdf[[c]].to_numpy(dtype=np.float64), y).mean()
            uni[c] = float(a)
        for c in SYN:
            per_q_rows.append(dict(dataset=name, qid=c, sense=QMETA[c]["sense"],
                                   set=QMETA[c]["set"], predicate=QMETA[c]["predicate"],
                                   metaphoricity=QMETA[c]["a_priori_metaphoricity"],
                                   uni_acc=uni[c], uni_acc_over_chance=uni[c] - chance))

        # Per-sense (low-poly only) group accuracy
        per_sense = {}
        for s in SENSES:
            cols = [q["qid"] for q in QRECS if q["sense"] == s and q["set"] == "SYN_LOW"]
            per_sense[s] = float(acc_cv(fdf[cols].to_numpy(dtype=np.float64), y).mean())

        # Incremental greedy curves
        curves = {}
        for g, cols in [("SYN_LOW", LOW), ("SYN_HIGH", HIGH)]:
            X = fdf[cols].to_numpy(dtype=np.float64)
            curves[g] = [(k, float(a), c) for (k, a, c) in greedy_forward(X, y, cols)]

        # Metaphoricity vs utility (this dataset)
        meta_arr = np.array([QMETA[c]["a_priori_metaphoricity"] for c in SYN])
        util_arr = np.array([uni[c] - chance for c in SYN])
        pr = pearsonr(meta_arr, util_arr); sr = spearmanr(meta_arr, util_arr)

        results["datasets"][name] = dict(
            n=len(y), n_classes=int(nclass), chance=float(chance),
            groups=grp, per_sense=per_sense, curves=curves,
            metaphoricity_corr=dict(pearson_r=float(pr[0]), pearson_p=float(pr[1]),
                                    spearman_r=float(sr.correlation), spearman_p=float(sr.pvalue)),
            uni=uni,
        )

    # ---- aggregate metaphoricity analysis across datasets ----
    pq = pd.DataFrame(per_q_rows)
    pq.to_csv(os.path.join(RES, "per_question.csv"), index=False)
    if len(pq):
        # mean over-chance utility per question (averaged across datasets)
        agg = pq.groupby(["qid", "set", "sense", "predicate", "metaphoricity"], as_index=False)["uni_acc_over_chance"].mean()
        pr = pearsonr(agg["metaphoricity"], agg["uni_acc_over_chance"])
        sr = spearmanr(agg["metaphoricity"], agg["uni_acc_over_chance"])
        results["metaphoricity_overall"] = dict(pearson_r=float(pr[0]), pearson_p=float(pr[1]),
                                                 spearman_r=float(sr.correlation), spearman_p=float(sr.pvalue),
                                                 low_mean=float(agg[agg.set == "SYN_LOW"]["uni_acc_over_chance"].mean()),
                                                 high_mean=float(agg[agg.set == "SYN_HIGH"]["uni_acc_over_chance"].mean()))

        # ---- FIGURE 1: metaphoricity vs utility scatter ----
        plt.figure(figsize=(5.2, 4.0))
        for st, mk, lab in [("SYN_LOW", "o", "Low-polysemy"), ("SYN_HIGH", "^", "High-polysemy")]:
            s = agg[agg.set == st]
            plt.scatter(s["metaphoricity"], s["uni_acc_over_chance"], marker=mk, label=lab, alpha=0.8)
        z = np.polyfit(agg["metaphoricity"], agg["uni_acc_over_chance"], 1)
        xs = np.linspace(0, 1, 50)
        plt.plot(xs, np.polyval(z, xs), "k--", lw=1,
                 label=f"fit (r={pr[0]:+.2f}, p={pr[1]:.2f})")
        plt.xlabel("A-priori metaphoricity of the sensory question")
        plt.ylabel("Mean accuracy over chance (univariate)")
        plt.title("Feature usefulness vs metaphoricity")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "metaphoricity_scatter.png"), dpi=160); plt.close()

    # ---- FIGURE 2: group accuracy bars per dataset ----
    if results["datasets"]:
        names = list(results["datasets"].keys())
        groups = ["GIB", "SYN_HIGH", "SYN_LOW", "SYN_ALL", "TFIDF"]
        plt.figure(figsize=(7.2, 4.0))
        w = 0.16
        x = np.arange(len(names))
        for gi, g in enumerate(groups):
            vals = [results["datasets"][n]["groups"].get(g, {}).get("acc", np.nan) for n in names]
            plt.bar(x + gi * w, vals, w, label=g)
        ch = [results["datasets"][n]["chance"] for n in names]
        for xi, c in zip(x, ch):
            plt.hlines(c, xi - w, xi + 5 * w, colors="gray", linestyles=":", lw=1)
        plt.xticks(x + 2 * w, names); plt.ylabel("5-fold CV accuracy")
        plt.title("Classification accuracy by feature set (dotted = chance)")
        plt.legend(fontsize=8, ncol=5); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "group_accuracy.png"), dpi=160); plt.close()

        # ---- FIGURE 3: incremental greedy curves (one subplot per dataset) ----
        nd = len(names)
        fig, axes = plt.subplots(1, nd, figsize=(3.2 * nd, 3.2), sharey=True)
        if nd == 1:
            axes = [axes]
        for ax, n in zip(axes, names):
            for g, style in [("SYN_LOW", "o-"), ("SYN_HIGH", "^--")]:
                cur = results["datasets"][n]["curves"][g]
                ax.plot([k for k, _, _ in cur], [a for _, a, _ in cur], style, ms=3, label=g)
            ax.hlines(results["datasets"][n]["chance"], 1, 20, colors="gray", linestyles=":")
            ax.set_title(n); ax.set_xlabel("# features"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("CV accuracy"); axes[0].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "incremental_curves.png"), dpi=160); plt.close()

        # ---- FIGURE 4: per-sense (low-poly) heatmap ----
        sense_mat = np.array([[results["datasets"][n]["per_sense"][s] - results["datasets"][n]["chance"]
                               for s in SENSES] for n in names])
        plt.figure(figsize=(5.6, 3.6))
        im = plt.imshow(sense_mat, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="acc over chance")
        plt.xticks(range(len(SENSES)), SENSES); plt.yticks(range(len(names)), names)
        plt.title("Per-sense informativeness (low-polysemy, 4 feats each)")
        for i in range(len(names)):
            for j in range(len(SENSES)):
                plt.text(j, i, f"{sense_mat[i,j]:+.2f}", ha="center", va="center",
                         color="w", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "per_sense_heatmap.png"), dpi=160); plt.close()

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote results.json, per_question.csv, and figures.")
    if "metaphoricity_overall" in results:
        m = results["metaphoricity_overall"]
        print(f"\nKEY: metaphoricity vs utility  pearson r={m['pearson_r']:+.3f} (p={m['pearson_p']:.3f}); "
              f"low_mean={m['low_mean']:+.3f}  high_mean={m['high_mean']:+.3f}")


if __name__ == "__main__":
    main()
