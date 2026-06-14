"""Metaphoricity-vs-utility analysis using the INDEPENDENT LLM metaphoricity measure.

Combines results/per_question.csv (univariate utility per dataset) with
results/metaphoricity_llm.csv. Produces:
  results/metaphoricity_analysis.json
  figures/metaphoricity_llm_scatter.png
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

pq = pd.read_csv(os.path.join(RES, "per_question.csv"))
llm = pd.read_csv(os.path.join(RES, "metaphoricity_llm.csv"))[["qid", "llm_metaphoricity_logit"]]

# mean univariate over-chance utility per question, averaged across datasets
agg = pq.groupby(["qid", "set", "sense", "predicate", "metaphoricity"], as_index=False)["uni_acc_over_chance"].mean()
agg = agg.merge(llm, on="qid", how="left")

# gibberish reference: mean univariate over-chance utility for GIB across datasets
res = json.load(open(os.path.join(RES, "results.json")))
gib_over_chance = []
for name, d in res["datasets"].items():
    ch = d["chance"]
    for qid, a in d["uni"].items():
        if qid.startswith("gib_"):
            gib_over_chance.append(a - ch)
gib_ref = float(np.mean(gib_over_chance))

out = {"gibberish_mean_over_chance": gib_ref}
for mcol, label in [("metaphoricity", "a_priori"), ("llm_metaphoricity_logit", "llm")]:
    pr = pearsonr(agg[mcol], agg["uni_acc_over_chance"])
    sr = spearmanr(agg[mcol], agg["uni_acc_over_chance"])
    out[label] = dict(pearson_r=float(pr[0]), pearson_p=float(pr[1]),
                      spearman_r=float(sr.correlation), spearman_p=float(sr.pvalue))

low = agg[agg.set == "SYN_LOW"]["uni_acc_over_chance"]
high = agg[agg.set == "SYN_HIGH"]["uni_acc_over_chance"]
u = mannwhitneyu(low, high, alternative="two-sided")
out["low_vs_high"] = dict(low_mean=float(low.mean()), high_mean=float(high.mean()),
                          low_min=float(low.min()), gib_mean=gib_ref,
                          mannwhitney_p=float(u.pvalue),
                          low_all_above_gib=bool((low > gib_ref).all()))

json.dump(out, open(os.path.join(RES, "metaphoricity_analysis.json"), "w"), indent=2)
print(json.dumps(out, indent=2))

# Figure: LLM metaphoricity vs utility
plt.figure(figsize=(5.4, 4.0))
for st, mk, c, lab in [("SYN_LOW", "o", "#1f77b4", "Low-polysemy (20)"),
                       ("SYN_HIGH", "^", "#d62728", "High-polysemy (20)")]:
    s = agg[agg.set == st]
    plt.scatter(s["llm_metaphoricity_logit"], s["uni_acc_over_chance"], marker=mk,
                c=c, alpha=0.8, label=lab)
z = np.polyfit(agg["llm_metaphoricity_logit"], agg["uni_acc_over_chance"], 1)
xs = np.linspace(agg["llm_metaphoricity_logit"].min(), agg["llm_metaphoricity_logit"].max(), 50)
pr = pearsonr(agg["llm_metaphoricity_logit"], agg["uni_acc_over_chance"])
plt.plot(xs, np.polyval(z, xs), "k--", lw=1, label=f"fit (r={pr[0]:+.2f})")
plt.axhline(gib_ref, color="gray", ls=":", lw=1.2, label=f"gibberish baseline ({gib_ref:+.3f})")
plt.xlabel("LLM metaphoricity of the sensory predicate (logit)")
plt.ylabel("Mean accuracy over chance (univariate)")
plt.title("Utility rises with metaphoricity, yet low-polysemy\nsensory probes remain informative (mostly above baseline)")
plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "metaphoricity_llm_scatter.png"), dpi=160); plt.close()
print("saved figures/metaphoricity_llm_scatter.png")
