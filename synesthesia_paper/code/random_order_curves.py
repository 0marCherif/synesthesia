"""Incremental accuracy under RANDOM feature orderings (10 permutations).

For each dataset and each k=1..K, we add features in a random order and measure
5-fold CV accuracy with the first k features, averaged over N_PERM random orderings.
A steadily rising mean (with a shrinking gap to the full-set accuracy) shows that, on
average, each additional synesthetic feature contributes new information.

Outputs:
  figures/random_order_curves.png
  results/random_order_curves.json
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from questions import as_records  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA, FEAT, RES, FIG = (os.path.join(ROOT, d) for d in ["data", "features", "results", "figures"])

DATASETS = ["imdb", "agnews", "bbc", "emotion"]
QRECS = as_records()
SYN = [q["qid"] for q in QRECS if q["set"] == "SYN_LOW"]  # 20 low-polysemy synesthetic features
N_PERM = 10
SEED = 0
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    K_FIXED = len(SYN)
    axes = axes.ravel()
    for ax, name in zip(axes, DATASETS):
        fp = os.path.join(FEAT, f"{name}_features.parquet")
        fdf = pd.read_parquet(fp)
        fdf[SYN] = fdf[SYN].astype("float64")
        X = fdf[SYN].to_numpy(dtype=np.float64)
        y = np.asarray(fdf["label"], dtype=np.int64)
        chance = float(pd.Series(y).value_counts(normalize=True).max())
        K = X.shape[1]

        curves = np.zeros((N_PERM, K))
        for p in range(N_PERM):
            perm = rng.permutation(K)
            for k in range(1, K + 1):
                cols = perm[:k]
                curves[p, k - 1] = cross_val_score(clf(), X[:, cols], y, cv=CV, scoring="accuracy").mean()
            print(f"{name}: perm {p+1}/{N_PERM} done", flush=True)
        mean = curves.mean(0); std = curves.std(0)
        ks = np.arange(1, K + 1)
        ax.plot(ks, mean, "-", color="#1f77b4", lw=1.5, label="random order (mean of 10)")
        ax.fill_between(ks, mean - std, mean + std, color="#1f77b4", alpha=0.25, label="$\\pm$1 s.d.")
        ax.hlines(chance, 1, K, colors="gray", linestyles=":", lw=1, label="chance")
        ax.set_title(f"{name}  (20 feats: {mean[-1]:.2f})", fontsize=10)
        ax.set_xlim(1, K_FIXED); ax.set_xticks(range(0, K_FIXED + 1, 4))
        ax.grid(alpha=0.3)
        out[name] = dict(k=ks.tolist(), mean=mean.tolist(), std=std.tolist(), chance=chance)
    for ax in axes[2:]:
        ax.set_xlabel("# features added (random order)")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("5-fold CV accuracy")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Accuracy grows steadily as features are added in random order\n"
                 "(20 low-polysemy synesthetic features, 10 random orderings)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "random_order_curves.png"), dpi=160)
    plt.close(fig)
    json.dump(out, open(os.path.join(RES, "random_order_curves.json"), "w"), indent=2)

    # report mean marginal gain (monotonicity evidence)
    print("\nMean marginal gain per added feature (averaged over permutations):")
    for name in DATASETS:
        m = np.array(out[name]["mean"])
        diffs = np.diff(m)
        print(f"  {name:8s} first->full: {m[0]:.3f}->{m[-1]:.3f}  "
              f"mean step={diffs.mean():+.4f}  frac steps>0={np.mean(diffs>0):.2f}")
    print("saved figures/random_order_curves.png")


if __name__ == "__main__":
    main()
