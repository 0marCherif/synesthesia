# Experiment Log — LLM Synesthesia / Absent Referent (short paper)

Running log of every experiment, decision, and result. Newest entries appended at the bottom.

## Project thesis
Pose synesthetic questions (5 senses) to an LLM about a text, extract `logit(yes) - logit(no)`
as features, show they classify across datasets, and argue (Saussure) that because an LLM has no
referent, sensory predicates are unconstrained signifiers — so even the *least metaphorical*
sensory questions still carry classificatory information.

Key manipulated variable: **metaphoricity / polysemy** of each sensory question.
Prediction: feature usefulness is (largely) independent of metaphoricity.

## Environment
- Host: linux 6.8, 4 vCPU, 32 GB RAM, 98 GB disk
- GPU: 1x NVIDIA A100-SXM4-80GB, CUDA 13.0, driver 580.126.20
- Python 3.12.13; torch 2.11.0; transformers 5.12.0; datasets 5.0.0; scikit-learn 1.8.0
- Internet: available. HF: apache-2.0 Gemma 4 models (ungated).

## Model
- `google/gemma-4-E4B-it` (Gemma 4, ~8B params, architecture `gemma4`, apache-2.0, released ~Jun 2026).
- Rationale: recent ("Gemma 4" as requested), light enough for ~100k forward passes, well-supported.

## Datasets (planned)
- IMDB (sentiment, binary)
- AG News (topic, 4-class)
- BBC News (topic, 5-class) — mirrors the original paper's 5-genre task
- Emotion / dair-ai/emotion (affect, 6-class)
- (optional) Quora insincere questions (sincerity) to connect to the source paper

Each: stratified balanced subset of N=500 (configurable), texts truncated.

## Question sets
- SYN-LOW: 20 low-polysemy synesthetic questions (4 per sense)
- SYN-HIGH: 20 high-polysemy (lexicalized-metaphor) synesthetic questions (4 per sense)
- GIB: 10 gibberish/random-word control "questions"
- Feature = logit(yes) - logit(no) at the answer slot.

---

## Timeline

### 2026-06-14 — Setup
- Probed environment (A100 80GB, internet OK). Installed transformers/datasets/accelerate.
- Chose model google/gemma-4-E4B-it (apache-2.0, ungated).
- Created project structure under /home/ubuntu/synesthesia_paper.

### 2026-06-14 — Model smoke test (code/smoke_test.py)
- Loads via AutoModelForCausalLM -> Gemma4ForConditionalGeneration, bf16 on cuda. Load ~117s, 17.6 GB VRAM.
- yes ids {4443,8438,10784,11262,26915,51327}; no ids {951,1904,2301,3771,7018,9424}.
  Aggregate via logsumexp over case/space variants.
- Feature = logsumexp(yes logits) - logsumexp(no logits) at answer slot (chat template, add_generation_prompt).
- Sanity: model is saturated toward "no" but the logit difference VARIES by question
  (smoky -21.8, purple -19.0, salty -20.9, loud -13.1) -> usable continuous signal.
  TODO at extraction time: confirm variation ACROSS texts for a fixed question.
- Throughput: batch=32, seqlen 48 -> ~232 seq/s. ~100k passes feasible in <20 min.
- Decisions: N=500 balanced/stratified per dataset; truncate text to ~300 words.

### 2026-06-14 — Question sets (code/questions.py)
- SYN_LOW (20): low-metaphoricity sensory predicates (orange, purple, umami, starchy, grainy,
  spongy, muffled, echoey, nasal, metallic, smoky, floral, ...). a_priori metaphoricity ~0.05-0.30.
- SYN_HIGH (20): lexicalized-metaphor predicates (dark, bright, sweet, bitter, warm, cold, rough,
  smooth, loud, shrill, stinking, fishy, fresh, ...). a_priori metaphoricity ~0.75-0.95.
- GIB (10): random-word "evoke these words" probes (control: does any probe work?).

### 2026-06-14 — Feature extraction (code/extract_features.py)
- Note: a first launch was killed when the parent shell was interrupted; relaunched detached via setsid.
- Feature = logsumexp(yes logits) - logsumexp(no logits) at answer slot; left-padding; dynamic batching (24k token budget).
- Throughput: short datasets ~300-420 prompts/s, imdb ~111/s, bbc slower (longer texts). GPU ~68 GB at full batch.
- Cross-text std per question healthy (agnews median 2.8; emotion median 3.4) -> real per-text signal, not constant.
- Outputs: features/<name>_features.parquet (50 feature cols + label + label_text).

### 2026-06-14 — Analysis pass 1 (partial: agnews + emotion) (code/analyze.py)
- Classifier: StandardScaler + LogisticRegression, 5-fold stratified CV. Baselines: GIB, TF-IDF(1-2gram,5k).
- agnews (4-class, chance .25): SYN_LOW .658, SYN_HIGH .662, SYN_ALL .710, GIB .558, TFIDF .752.
- emotion (6-class, chance .167): SYN_LOW .287, SYN_HIGH .390, SYN_ALL .424, GIB .235, TFIDF .425.
- KEY (2 datasets so far): metaphoricity vs univariate over-chance utility pearson r=+0.27 (p=.09);
  low_mean=+.047, high_mean=+.061. => low-polysemy ~ as useful as high-polysemy; both >> gibberish/chance.
  Metaphor gives a task-dependent boost (strong for affective 'emotion', ~none for topic 'agnews').
- Note pandas 3.0 arrow-backed arrays broke sklearn indexing -> convert with .to_numpy(float)/np.asarray.
- (Will rerun with all 4 datasets once imdb+bbc finish.)

### 2026-06-14 — Extraction complete (all 4 datasets), total ~1360s (~23 min)
- imdb 437s, agnews 137s, emotion 85s, bbc 589s. All cross-text stds healthy.

### 2026-06-14 — Analysis pass 2 (ALL 4 datasets) — FINAL numbers
Accuracy (5-fold CV, LogReg on standardized features); chance in parens:
| dataset           | chance | SYN_LOW | SYN_HIGH | SYN_ALL | GIB  | TFIDF |
|-------------------|--------|---------|----------|---------|------|-------|
| imdb (2, sent)    | .500   | .880    | .934     | .924    | .660 | .792  |
| agnews (4, topic) | .250   | .658    | .662     | .710    | .558 | .752  |
| bbc (5, topic)    | .200   | .714    | .654     | .768    | .504 | .966  |
| emotion (6, affect)| .167  | .287    | .390     | .424    | .235 | .425  |
Takeaways:
- Synesthetic feature SETS >> gibberish on EVERY dataset; >> chance everywhere.
- Beat TF-IDF on imdb (.92 vs .79) and tie on emotion; TF-IDF wins on topic (esp bbc .97).
- Low vs high polysemy is task-dependent: HIGH>LOW for affective (imdb,emotion); LOW>HIGH for bbc topic; ~equal agnews.

### 2026-06-14 — Independent LLM metaphoricity (code/metaphoricity_rating.py)
- Asked Gemma (yes/no logit) whether each predicate is "often used figuratively/metaphorically".
- Clean separation: SYN_HIGH strongly + (warm +11.7, dark +11.4, sweet +10.9, bitter +11.1);
  SYN_LOW mostly - (starchy -7.3, minty -6.2, nasal -6.0). Independent corrections: opaque +5.9,
  fuzzy +4.2, salty +2.7 ("salty"=annoyed) flagged as more metaphorical than our a-priori.

### 2026-06-14 — Metaphoricity vs utility (code/metaphoricity_analysis.py)
- Univariate over-chance utility (avg across datasets) vs metaphoricity:
  a_priori pearson r=+0.56 (p=2e-4); LLM-rated pearson r=+0.49 (p=1e-3). => metaphor helps individual features.
- low_mean over-chance=+0.070, high_mean=+0.112, gibberish_mean=+0.058 (Mann-Whitney low vs high p=5e-4).
- HONEST framing: metaphoricity correlates with single-feature utility, BUT (1) low-polysemy SETS
  beat gibberish/chance on every task, (2) all 5 senses are informative (per_sense_heatmap), so the
  effect does NOT require lexicalized figurative meaning -> supports the Saussurean "no referent" reading.

### Figures (figures/)
- group_accuracy.png (main results bars)
- metaphoricity_llm_scatter.png (key metaphoricity-vs-utility, w/ gibberish baseline line)
- incremental_curves.png (greedy forward selection, mirrors source paper figs)
- per_sense_heatmap.png (each sense informative on each dataset)

### 2026-06-14 — Paper written & compiled
- Title: "A Text Has No Taste: Synesthetic Probing of Language Models and the Absent Referent".
- paper/acl_latex.tex (ACL template). Added refs to paper/custom.bib (Saussure, Bender&Koller, Bender+2021,
  Harnad, Bisk, Firth, Winter, Strik Lievers, Ullmann, Lynott(Lancaster), Derrida, Gemma4).
- Renamed bib to ASCII mybib.bib (bibtex couldn't open accented "Mabibliothèque.bib"); \bibliography{mybib,custom}.
- Installed TeX Live + poppler. Compiles clean: 5 pages, 0 undefined citations. PDF: paper/acl_latex.pdf.
- Body ~3.5 pages (within ACL short-paper limit; Limitations + refs + appendix excluded).

### 2026-06-14 — Added random-order incremental curves (code/random_order_curves.py)
- Per dataset, add the 40 synesthetic features in RANDOM order, 10 permutations, 5-fold CV; plot mean +/- 1 s.d.
- Steady monotone-ish rise to full set; fraction of single-feature steps that improve accuracy:
  bbc 0.97, agnews 0.85, emotion 0.82, imdb 0.67 (imdb saturates early). first->full:
  imdb .70->.92, agnews .33->.71, bbc .31->.77, emotion .18->.42.
- => features are complementary (each adds new info, not redundant copies of one axis).
- New figure figures/random_order_curves.png; replaced greedy incremental_curves as paper Fig 4 (full-width figure*).
- Recompiled: still 5 pages, 0 undefined citations.

### 2026-06-14 — Random-order curves restricted to LOW-polysemy set (per request)
- Changed random_order_curves.py to use only the 20 SYN_LOW features; all 4 subplots share x-axis 1->20
  (set_xlim/xticks fixed). first->full and frac of positive steps (20 low-poly feats):
  imdb .59->.88 (89%), agnews .33->.66 (100%), bbc .31->.71 (95%), emotion .19->.29 (79%).
- Updated paper Fig 4 caption + "Features are complementary" paragraph with these numbers. 5 pages, 0 undefined.
- Note: subplot x-axes were already shared before; now fixed to 1-20. y-axes intentionally per-task
  (different chance levels and score ranges).

### 2026-06-14 — Zero-shot vs trained features, + MLP check (code/zero_shot_and_mlp.py)
- CLARIFICATION: the trained classifier is LOGISTIC REGRESSION (linear), not an MLP. Chosen as a weak
  classifier so accuracy reflects feature information. (Source paper used an MLP.)
- Zero-shot: closed-set MULTIPLE-CHOICE probe (options A/B/C/D...), read the option-letter logit,
  argmax (one forward/text). Robust; replaced an earlier label-logprob scoring that degenerated
  (agnews -> exactly chance .25 due to multi-token label bias).
- Results (acc): zero-shot | low(lr/mlp) | all(lr/mlp); chance in ():
  imdb (.50): zs .946 | .880/.868 | .924/.926
  agnews(.25): zs .842 | .658/.656 | .710/.704
  bbc (.20): zs .922 | .714/.708 | .768/.778
  emotion(.17): zs .496 | .287/.299 | .424/.372
- Takeaways: zero-shot (sees the task) is the upper reference; synesthetic features (never mention the
  task) recover most of it (IMDB within 2 pts). LR ~= MLP everywhere => signal is in features, not classifier.
- Added paper Table 2 (zero-shot vs trained, lr & mlp) + "Comparison with zero-shot" paragraph; noted MLP
  equivalence in setup. results/comparison.json. 5 pages, 0 undefined. (Body sections 1-6 ~4 pages; the
  full-width Fig 4 pushes Conclusion to top of p5 -- can tighten if strict 4-page body is required.)

## How to reproduce
1. python3 code/data_prep.py            # build data/*.parquet
2. python3 code/extract_features.py     # build features/*.parquet  (~23 min on A100)
3. python3 code/metaphoricity_rating.py # results/metaphoricity_llm.csv
4. python3 code/analyze.py              # results/*, figures/*
5. python3 code/metaphoricity_analysis.py
6. python3 code/random_order_curves.py   # figures/random_order_curves.png
7. python3 code/zero_shot_and_mlp.py      # results/comparison.json (zero-shot + MLP)
8. cd paper && pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex
