"""
STAT 451 Group Project -- Breast Cancer Wisconsin (Diagnostic)
Amoolya, Bonnie, Dhruvika, Kalynn, Edward

Question: Which nuclear-morphology measurements distinguish malignant from
benign breast masses, and how few of them do we actually need?

Protocol (this is the part the proposal got wrong, now fixed):
  * ONE stratified 80/20 train/test split.
  * ALL model choice, hyperparameter tuning, feature ranking and threshold
    selection happen by 5-fold stratified CV *inside the training set*.
  * The test set is touched in ONE pass at the very end, scoring the two models
    fixed before that pass: the preferred full-feature model, and the reduced
    model on K named features. Nothing is re-tuned after the test set is opened.

Data: Wolberg, Mangasarian, Street & Street (1993), Breast Cancer Wisconsin
(Diagnostic), UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
"""
# %%
import warnings
# Only silence the two we have checked and accept. A blanket ignore would also
# hide ConvergenceWarning, and then we could not tell a converged fit from one
# that quietly ran out of iterations.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score)

import os

SEED = 451
rng = np.random.default_rng(SEED)
OUT = {}                      # numbers the report/slides quote
FIGDIR = "figures"
CACHE = "wdbc.csv"            # so a rerun does not depend on UCI being up
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------- style ----
# Validated palette (dataviz six-checks, light mode, all-pairs: CVD dE 21.6).
BENIGN, MALIG = "#2a78d6", "#e34948"      # categorical slots: blue / red
SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_SEQ = LinearSegmentedColormap.from_list("blueseq", SEQ)
# diverging: benign pole <- neutral gray -> malignant pole
DIVERGE = LinearSegmentedColormap.from_list("bg r", [BENIGN, "#f0efec", MALIG])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False,
})
def pretty(c):
    return c.replace("_", " ").replace("concave points", "concave pts")

# ============================================================== 1. DATA ====
# %%
FEATS = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
         "concavity", "concave_points", "symmetry", "fractal_dimension"]
COLS = [f"{s}_{f}" for s in ["mean", "se", "worst"] for f in FEATS]
# Which of the ten measurements describe the same biological thing. Used both
# to summarise the chosen feature set and to pick the panels of Figure 2.
GROUPS = {"Size": ["radius", "perimeter", "area"],
          "Shape": ["compactness", "concavity", "concave_points"],
          "Texture": ["texture"],
          "Smoothness": ["smoothness", "symmetry", "fractal_dimension"]}
GROUP_OF = {f: g.lower() for g, fs in GROUPS.items() for f in fs}

if os.path.exists(CACHE):
    raw = pd.read_csv(CACHE)
else:
    wdbc = fetch_ucirepo(id=17)
    raw = pd.concat([wdbc.data.features, wdbc.data.targets], axis=1)
    raw.to_csv(CACHE, index=False)
X = raw.drop(columns=["Diagnosis"])
# UCI ships the ten measurements in three blocks suffixed 1/2/3 (mean, standard
# error, worst). We rename positionally, so verify the order first -- a silent
# reordering upstream would mislabel every feature in the report.
assert list(X.columns) == [f"{f}{i}" for i in (1, 2, 3) for f in FEATS], \
    f"unexpected UCI column order: {list(X.columns)}"
X.columns = COLS
y = raw["Diagnosis"].map({"M": 1, "B": 0}).values

OUT["n"], OUT["p"] = X.shape
OUT["n_malignant"] = int(y.sum())
OUT["n_benign"] = int((1 - y).sum())
OUT["pct_malignant"] = round(100 * y.mean(), 1)
OUT["baseline_acc"] = round(100 * (1 - y.mean()), 1)   # always-predict-benign
print(f"{OUT['n']} masses x {OUT['p']} features; "
      f"{OUT['n_malignant']} malignant ({OUT['pct_malignant']}%), "
      f"{OUT['n_benign']} benign. Always-benign baseline = {OUT['baseline_acc']}%")
print("Missing values:", int(X.isna().sum().sum()))

# ---- the one and only split ----
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                      random_state=SEED)
OUT["n_train"], OUT["n_test"] = len(Xtr), len(Xte)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ====================================== 2. REDUNDANCY AMONG THE FEATURES ===
# %%
corr = Xtr.corr()
off = corr.abs().where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
OUT["pairs_r90"] = int((off > 0.90).sum())
OUT["n_pairs"] = int(len(off))
OUT["max_r"] = round(float(off.max()), 3)
top_pair = off.idxmax()
OUT["max_r_pair"] = f"{pretty(top_pair[0])} & {pretty(top_pair[1])}"
print(f"{OUT['pairs_r90']} of {OUT['n_pairs']} feature pairs have |r|>0.90; "
      f"largest r={OUT['max_r']} ({OUT['max_r_pair']})")

# Fig 1: redundancy map. Job = magnitude (how redundant) -> ONE-hue sequential.
order = [f"{s}_{f}" for f in FEATS for s in ["mean", "se", "worst"]]
M = corr.loc[order, order].abs().values
fig, ax = plt.subplots(figsize=(8.4, 7.2))
im = ax.imshow(M, cmap=BLUE_SEQ, vmin=0, vmax=1)
ax.set_xticks(range(1, 30, 3)); ax.set_yticks(range(1, 30, 3))
ax.set_xticklabels([pretty(f) for f in FEATS], rotation=45, ha="right")
ax.set_yticklabels([pretty(f) for f in FEATS])
for k in range(3, 30, 3):
    ax.axhline(k - .5, color=SURFACE, lw=2); ax.axvline(k - .5, color=SURFACE, lw=2)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("|correlation|", color=INK2); cb.outline.set_visible(False)
ax.set_title("Each measurement appears three times, and the\nsize "
             "measurements repeat each other", loc="left", pad=14)
fig.text(0.02, 0.02, f"{OUT['pairs_r90']} of {OUT['n_pairs']} pairs exceed "
         "|r| = 0.90.  Dark blocks mean two features carry nearly the same "
         "information. Each block is mean / se / worst.",
         fontsize=10.5, color=INK2)
fig.savefig(f"{FIGDIR}/fig1_redundancy.png"); plt.close(fig)

# ------------------------------------- greedy de-duplication of features ---
# Rank by univariate discrimination, then keep a feature only if it is not
# already represented (|r| < 0.80 with everything kept so far).
uni_auc = pd.Series({c: roc_auc_score(ytr, Xtr[c]) for c in COLS}
                    ).sort_values(ascending=False)
# Four features run the other way (lower value = malignant), so raw AUC would
# rank them as useless. Discriminative power is |AUC - 0.5|, i.e. direction
# does not matter -- rank and select on that, but report the raw AUC.
uni_pow = uni_auc.apply(lambda a: max(a, 1 - a)).sort_values(ascending=False)
OUT["inverse_features"] = [(pretty(c), round(v, 3))
                           for c, v in uni_auc[uni_auc < 0.5].sort_values().items()]

kept = []
for c in uni_pow.index:
    if all(abs(corr.loc[c, k]) < 0.80 for k in kept):
        kept.append(c)
OUT["uni_top"] = [(pretty(c), round(v, 3)) for c, v in uni_auc.head(6).items()]
OUT["distinct_features"] = [pretty(c) for c in kept[:8]]
OUT["n_distinct"] = len(kept)
print("\nTop univariate AUC:", OUT["uni_top"])
print("Runs the other way (AUC < 0.5):", OUT["inverse_features"])
print(f"Distinct (|r|<0.80) representatives ({len(kept)} total):",
      OUT["distinct_features"])

# Does the summary type matter? Average discriminative power by group.
OUT["group_power"] = {g: round(float(uni_pow[[f"{g}_{f}" for f in FEATS]].mean()), 3)
                      for g in ["mean", "se", "worst"]}
print("Average discriminative power by summary type:", OUT["group_power"])

# ================================ 3. MODEL COMPARISON -- CV ON TRAIN ONLY ==
# %%
def pipe(model, scale=True, pca=None):
    steps = []
    if scale: steps.append(("sc", StandardScaler()))
    if pca:   steps.append(("pca", PCA(n_components=pca, random_state=SEED)))
    steps.append(("m", model))
    return Pipeline(steps)

SPACE = {
    "Logistic (L2)":  (pipe(LogisticRegression(max_iter=5000)),
                       {"m__C": [0.01, 0.1, 1, 10, 100]}),
    "Logistic (L1)":  (pipe(LogisticRegression(penalty="l1", solver="liblinear",
                                               max_iter=5000, random_state=SEED)),
                       {"m__C": [0.01, 0.05, 0.1, 0.5, 1, 10]}),
    "Decision tree":  (pipe(DecisionTreeClassifier(random_state=SEED), scale=False),
                       {"m__max_depth": [2, 3, 4, 5, 6, None],
                        "m__min_samples_leaf": [1, 5, 10]}),
    "k-NN":           (pipe(KNeighborsClassifier()),
                       {"m__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                        "m__weights": ["uniform", "distance"]}),
    "SVM (linear)":   (pipe(SVC(kernel="linear", probability=True,
                                random_state=SEED)),
                       {"m__C": [0.01, 0.1, 1, 10]}),
    "SVM (RBF)":      (pipe(SVC(kernel="rbf", probability=True,
                                random_state=SEED)),
                       {"m__C": [0.1, 1, 10, 100],
                        "m__gamma": ["scale", 0.01, 0.05, 0.1]}),
    "PCA + logistic": (pipe(LogisticRegression(max_iter=5000), pca=10),
                       {"pca__n_components": [2, 5, 10, 15],
                        "m__C": [0.01, 0.1, 1, 10]}),
}

results, fitted = {}, {}
for name, (pl, grid) in SPACE.items():
    gs = GridSearchCV(pl, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    r = gs.cv_results_
    i = gs.best_index_
    results[name] = dict(auc=r["mean_test_score"][i], sd=r["std_test_score"][i],
                         params={k.split("__")[-1]: v
                                 for k, v in gs.best_params_.items()})
    fitted[name] = gs.best_estimator_
    print(f"{name:16s} CV AUC {results[name]['auc']:.4f} "
          f"(sd {results[name]['sd']:.4f})  {results[name]['params']}")

cvtab = (pd.DataFrame(results).T.sort_values("auc", ascending=False))
BEST = cvtab.index[0]
OUT["best_model"] = BEST
OUT["cv_table"] = [(n, round(float(r.auc), 4), round(float(r.sd), 4), str(r.params))
                   for n, r in cvtab.iterrows()]
OUT["best_cv_auc"] = round(float(cvtab.iloc[0].auc), 4)
OUT["worst_cv_auc"] = round(float(cvtab.iloc[-1].auc), 4)
OUT["auc_spread"] = round(OUT["best_cv_auc"] - OUT["worst_cv_auc"], 4)
OUT["best_params"] = results[BEST]["params"]
OUT["runner_up"] = cvtab.index[1]
OUT["runner_up_gap"] = round(float(cvtab.iloc[0].auc - cvtab.iloc[1].auc), 4)
# How many rivals are indistinguishable from the winner, and how far back the
# laggard is -- both quoted in the report, so neither gets typed by hand.
OUT["n_within_001"] = int((cvtab.auc.astype(float)
                           >= float(cvtab.iloc[0].auc) - 0.001).sum() - 1)
OUT["tree_auc"] = round(float(results["Decision tree"]["auc"]), 4)
print(f"\nBest by CV AUC: {BEST}")

# Fig 4: model comparison -- deliberately small, it is not the headline.
fig, ax = plt.subplots(figsize=(7.6, 3.6))
o = cvtab.iloc[::-1]
yy = np.arange(len(o))
ax.errorbar(o.auc.astype(float), yy, xerr=o.sd.astype(float), fmt="o",
            color=INK2, ecolor=GRIDC, elinewidth=6, markersize=9, capsize=0)
ax.plot(float(cvtab.iloc[0].auc), len(o) - 1, "o", color=MALIG, markersize=9,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
ax.set_yticks(yy); ax.set_yticklabels(o.index)
ax.set_xlabel("5-fold cross-validated ROC-AUC (training set)")
ax.set_title("All seven models land within 0.03 AUC of each other",
             loc="left", pad=10)
ax.text(0.5, -0.42, f"Spread from best to worst is {OUT['auc_spread']:.3f} AUC. "
        "Bars are ±1 SD across folds.", transform=ax.transAxes, ha="center",
        fontsize=10.5, color=INK2)
ax.grid(axis="y", visible=False)
fig.savefig(f"{FIGDIR}/fig4_models.png"); plt.close(fig)

# ============================= 4. WHICH FEATURES? (COLLINEARITY-AWARE) =====
# %%
# 4a. Naive read: L1 coefficients from a single fit.
l1 = fitted["Logistic (L1)"]
coef = pd.Series(l1.named_steps["m"].coef_[0], index=COLS)
OUT["l1_C"] = results["Logistic (L1)"]["params"]["C"]
OUT["l1_nonzero"] = int((coef != 0).sum())
print(f"A single L1 fit (C={OUT['l1_C']}) keeps {OUT['l1_nonzero']} of 30 features.")

# 4b. Honest read: is that set stable? Refit L1 on 200 bootstrap resamples.
B = 200
sel = pd.DataFrame(0, index=range(B), columns=COLS)
Xtr_v, ytr_v = Xtr.values, ytr
for b in range(B):
    idx = rng.choice(len(Xtr_v), len(Xtr_v), replace=True)
    # Redraw rather than skip: a skipped row would stay all-zero and count as a
    # resample that selected nothing, deflating every frequency.
    while len(np.unique(ytr_v[idx])) < 2:
        idx = rng.choice(len(Xtr_v), len(Xtr_v), replace=True)
    # liblinear shuffles internally -- seed it, or the stability numbers move
    # between runs and the analysis stops being reproducible.
    p = pipe(LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000,
                                C=OUT["l1_C"], random_state=SEED))
    p.fit(Xtr_v[idx], ytr_v[idx])
    sel.iloc[b] = (p.named_steps["m"].coef_[0] != 0).astype(int)
freq = (sel.mean() * 100).sort_values(ascending=False)
OUT["stability_top"] = [(pretty(c), round(v, 1)) for c, v in freq.head(10).items()]

size_block = ["mean_radius", "mean_perimeter", "mean_area",
              "worst_radius", "worst_perimeter", "worst_area"]
OUT["size_freq"] = {pretty(c): round(float(freq[c]), 1) for c in size_block}
OUT["size_any"] = round(float(sel[size_block].any(axis=1).mean() * 100), 1)
OUT["size_max"] = round(float(freq[size_block].max()), 1)

# The most stable feature is the one the report tells a story about, so export
# every number that story quotes instead of typing them into the prose by hand.
TEX = freq.index[0]
OUT["stable_first"] = pretty(TEX)
OUT["stable_first_freq"] = round(float(freq[TEX]), 1)
OUT["stable_first_rank"] = int(list(uni_pow.index).index(TEX) + 1)
OUT["stable_first_auc"] = round(float(uni_pow[TEX]), 2)
OUT["stable_first_r_size"] = round(float(corr.loc[TEX, size_block].abs().max()), 2)
print("\nBootstrap selection frequency (%), top 10:")
print(freq.head(10).round(1))
print(f"\nA size feature is selected in {OUT['size_any']}% of resamples, but no "
      f"single size feature exceeds {OUT['size_max']}%.")

# Fig 6: stability overall, plus the size block on its own so the splintering
# of one signal across six interchangeable columns is unmistakable.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
show = freq.head(12).iloc[::-1]
colors = [MALIG if c in size_block else BENIGN for c in show.index]
a1.barh(range(len(show)), show.values, color=colors, height=0.68)
a1.set_yticks(range(len(show)))
a1.set_yticklabels([pretty(c) for c in show.index])
a1.set_xlabel("% of 200 bootstrap resamples kept")
a1.set_xlim(0, 112)
for i, v in enumerate(show.values):
    a1.text(v + 2, i, f"{v:.0f}", va="center", fontsize=10.5, color=INK2)
a1.grid(axis="y", visible=False)
a1.set_title("Most stable features", loc="left", pad=8)

sb = freq[size_block].sort_values()
a2.barh(range(len(sb)), sb.values, color=MALIG, height=0.68)
a2.set_yticks(range(len(sb)))
a2.set_yticklabels([pretty(c) for c in sb.index])
a2.set_xlabel("% of 200 bootstrap resamples kept")
a2.set_xlim(0, 112)
for i, v in enumerate(sb.values):
    a2.text(v + 2, i, f"{v:.0f}", va="center", fontsize=10.5, color=INK2)
a2.axvline(OUT["size_any"], color=INK, lw=2, ls=(0, (4, 3)))
a2.text(OUT["size_any"] - 3, 0.1, f"at least one of\nthe six: {OUT['size_any']:.0f}%",
        ha="right", fontsize=10.5, color=INK)
a2.grid(axis="y", visible=False)
a2.set_title("The six size features split the vote", loc="left", pad=8)

fig.suptitle("LASSO always keeps a size feature, but not the same one",
             x=0.02, ha="left", fontsize=15, y=1.02)
rmin = corr.loc[size_block, size_block].values[np.triu_indices(6, 1)].min()
fig.text(0.02, -0.05, "Some size feature survives in "
         f"{OUT['size_any']:.0f}% of resamples, but which one is close to random, "
         f"because they\ncorrelate {rmin:.2f}–1.00 with each other. A single "
         "fitted coefficient would not have shown this.",
         fontsize=10.5, color=INK2)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig6_stability.png"); plt.close(fig)

# 4c. Third view: permutation importance, computed OUT OF FOLD. Permuting the
# same rows the model was fit on measures what it leaned on in sample, which
# flatters whatever it overfit. Refit per fold and permute held-out rows only.
perm_folds = []
for tr_i, va_i in cv.split(Xtr, ytr):
    est = clone(fitted[BEST]).fit(Xtr.iloc[tr_i], ytr[tr_i])
    pi = permutation_importance(est, Xtr.iloc[va_i], ytr[va_i], n_repeats=10,
                                random_state=SEED, scoring="roc_auc")
    perm_folds.append(pi.importances_mean)
perm = (pd.Series(np.mean(perm_folds, axis=0), index=COLS)
        .sort_values(ascending=False))
OUT["perm_top"] = [(pretty(c), round(v, 4)) for c, v in perm.head(8).items()]
OUT["perm_first"] = pretty(perm.index[0])
print("\nPermutation importance (out of fold), top 8:")
print(perm.head(8).round(4))

# ================================ 5. HOW FEW FEATURES DO WE NEED? ==========
# %%
# Selection happens INSIDE each fold, so the curve is not optimistically biased.
def curve_raw(k_values):
    out = []
    for k in k_values:
        sc = []
        for tr_i, va_i in cv.split(Xtr, ytr):
            Xa, Xb = Xtr.iloc[tr_i], Xtr.iloc[va_i]
            ya, yb = ytr[tr_i], ytr[va_i]
            a = pd.Series({c: roc_auc_score(ya, Xa[c]) for c in COLS}
                          ).apply(lambda v: max(v, 1 - v)).sort_values(ascending=False)
            cr = Xa.corr()
            keep = []
            for c in a.index:
                if all(abs(cr.loc[c, j]) < 0.80 for j in keep):
                    keep.append(c)
                if len(keep) == k:
                    break
            # Guard: if the |r|<0.80 filter cannot supply k features the curve
            # would silently flatten instead of erroring.
            assert len(keep) == k, f"only {len(keep)} features available for k={k}"
            p = pipe(LogisticRegression(max_iter=5000, C=1))
            p.fit(Xa[keep], ya)
            sc.append(roc_auc_score(yb, p.predict_proba(Xb[keep])[:, 1]))
        out.append(np.mean(sc))
    return np.array(out)

def curve_pca(k_values):
    out = []
    for k in k_values:
        sc = []
        for tr_i, va_i in cv.split(Xtr, ytr):
            Xa, Xb = Xtr.iloc[tr_i], Xtr.iloc[va_i]
            ya, yb = ytr[tr_i], ytr[va_i]
            p = pipe(LogisticRegression(max_iter=5000, C=1), pca=k)
            p.fit(Xa, ya)
            sc.append(roc_auc_score(yb, p.predict_proba(Xb)[:, 1]))
        out.append(np.mean(sc))
    return np.array(out)

ks = list(range(1, 13))
auc_raw, auc_pca = curve_raw(ks), curve_pca(ks)
# Reference must be the SAME model on all 30 features, or the comparison is
# tuned-30 against untuned-k and the reduced set looks worse than it is.
full_auc = float(np.mean(
    [roc_auc_score(ytr[va], pipe(LogisticRegression(max_iter=5000, C=1))
                   .fit(Xtr.iloc[tr], ytr[tr])
                   .predict_proba(Xtr.iloc[va])[:, 1])
     for tr, va in cv.split(Xtr, ytr)]))
OUT["auc_k"] = {k: round(float(v), 4) for k, v in zip(ks, auc_raw)}
OUT["auc_pca_k"] = {k: round(float(v), 4) for k, v in zip(ks, auc_pca)}
OUT["full30_auc"] = round(float(full_auc), 4)

# Smallest k whose CV AUC comes within TOL of all 30 features. TOL is a
# judgement call, and k=6 clears the 0.005 bar by well under a fold's worth of
# noise, so report the stricter answer too rather than let 6 look inevitable.
def smallest_k(tol):
    ok = [k for k, v in zip(ks, auc_raw) if v >= full_auc - tol]
    if not ok:
        raise RuntimeError(f"no k in {ks} lands within {tol} AUC of all 30")
    return ok[0]

K, K_STRICT = smallest_k(0.005), smallest_k(0.002)
OUT["K"], OUT["K_strict"] = K, K_STRICT
OUT["auc_at_K"] = round(float(auc_raw[ks.index(K)]), 4)
OUT["auc_at_K_strict"] = round(float(auc_raw[ks.index(K_STRICT)]), 4)
OUT["K_margin"] = round(float(auc_raw[ks.index(K)] - (full_auc - 0.005)), 4)
OUT["pca_gain_at_K"] = round(float(auc_pca[ks.index(K)] - auc_raw[ks.index(K)]), 4)
FINAL_FEATURES = kept[:K]
OUT["final_features"] = [pretty(c) for c in FINAL_FEATURES]
# e.g. "two of size, three of shape, one of texture"
NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
mix = pd.Series([GROUP_OF[c.split("_", 1)[1]] for c in FINAL_FEATURES]
                ).value_counts()
OUT["final_mix"] = ", ".join(f"{NUMWORD[n]} {g}" for g, n in mix.items())
print(f"\n{K} features reach CV AUC {OUT['auc_at_K']} vs {OUT['full30_auc']} "
      f"for all 30 (margin {OUT['K_margin']}; at tol 0.002 it would be "
      f"{K_STRICT}).\nChosen ({OUT['final_mix']}): {OUT['final_features']}")

# PCA scree, for the sentence in the report that has to introduce it
pca_full = PCA().fit(StandardScaler().fit_transform(Xtr))
cum = np.cumsum(pca_full.explained_variance_ratio_)
OUT["pc1_var"] = round(100 * pca_full.explained_variance_ratio_[0], 1)
OUT["pc12_var"] = round(100 * cum[1], 1)
OUT["pc_for_95"] = int(np.searchsorted(cum, 0.95) + 1)
print(f"PC1 alone captures {OUT['pc1_var']}% of variance; "
      f"{OUT['pc_for_95']} PCs are needed for 95%.")

# Fig 5: interpretable reduction vs PCA reduction.
fig, ax = plt.subplots(figsize=(7.8, 4.6))
ax.axhline(full_auc, color=GRIDC, lw=2, ls=(0, (4, 3)))
ax.text(12, full_auc + 0.004, "all 30 features", ha="right", fontsize=10.5,
        color=INK2)
ax.plot(ks, auc_raw, "-o", color=BENIGN, markersize=7,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label="Named features")
ax.plot(ks, auc_pca, "-o", color=MALIG, markersize=7,
        markeredgecolor=SURFACE, markeredgewidth=1.5,
        label="Principal components")
ax.plot(K, auc_raw[ks.index(K)], "o", color=BENIGN, markersize=16,
        markerfacecolor="none", markeredgewidth=2.5, zorder=5)
ax.annotate(f"{K} features, almost\nthe same AUC",
            xy=(K, auc_raw[ks.index(K)]),
            xytext=(K + 0.5, auc_raw[ks.index(K)] - 0.011),
            fontsize=11.5, color=INK, va="top",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2,
                            shrinkA=0, shrinkB=9))
ax.set_xlabel("Number of features (or components) kept")
ax.set_ylabel("Cross-validated ROC-AUC")
ax.set_xticks(ks); ax.legend(loc="lower right")
ax.set_title("Six features get most of what all thirty give", loc="left", pad=10)
gap = max(auc_pca[i] - auc_raw[i] for i in range(1, len(ks)))
OUT["pca_gain_max"] = round(float(gap), 4)
ax.text(0, -0.30, f"From two features on, components lead, but by at most "
        f"{gap:.3f} AUC, and no component is something a\npathologist can "
        "measure. Features are re-ranked inside every fold, so the curve is not "
        "optimistically biased.",
        transform=ax.transAxes, fontsize=10.5, color=INK2, va="top")
fig.savefig(f"{FIGDIR}/fig5_howfew.png"); plt.close(fig)

# ================== 6. WHAT MALIGNANCY LOOKS LIKE, ONE FEATURE AT A TIME ===
# %%
# P-hat(malignant | x_j), estimated without assuming a logistic shape: a
# Gaussian-kernel (Nadaraya-Watson) smoother, plus binned empirical rates.
def smooth_prob(x, lab, grid):
    bw = 1.2 * x.std() * len(x) ** (-1 / 5)          # Silverman-style bandwidth
    w = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2)
    return (w * lab).sum(1) / np.maximum(w.sum(1), 1e-9)

# One representative per biological group -- the most informative of each.
panel = []
for g, members in GROUPS.items():
    cands = [c for c in COLS if c.split("_", 1)[1] in members]
    panel.append((g, uni_pow[cands].idxmax()))
OUT["panel"] = [(g, pretty(c), round(float(uni_pow[c]), 3)) for g, c in panel]

fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
for ax, (grp, c) in zip(axes.ravel(), panel):
    xs = Xtr[c].values
    lo, hi = np.percentile(xs, [0.5, 99.5])
    gridx = np.linspace(lo, hi, 300)
    ax.plot(gridx, smooth_prob(xs, ytr, gridx), color=INK, lw=2.4, zorder=4)

    edges = np.quantile(xs, np.linspace(0, 1, 9))
    edges[-1] += 1e-9
    b = np.digitize(xs, edges[1:-1])
    for g in np.unique(b):
        s = b == g
        if s.sum() >= 8:
            ax.plot(xs[s].mean(), ytr[s].mean(), "o", color=MALIG, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)
    ax.plot(xs[ytr == 0], np.full((ytr == 0).sum(), -0.06), "|", color=BENIGN,
            alpha=.55, markersize=7)
    ax.plot(xs[ytr == 1], np.full((ytr == 1).sum(), 1.06), "|", color=MALIG,
            alpha=.55, markersize=7)
    ax.axhline(ytr.mean(), color=GRIDC, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(-0.13, 1.13)
    ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", ".5", "1"])
    ax.set_xlabel(pretty(c))
    ax.set_title(f"{grp}", loc="left", fontsize=12.5)
    # uni_pow is direction-corrected, so spell out when a feature runs the other
    # way rather than printing a number that reads like a plain AUC.
    lab = (f"AUC {uni_auc[c]:.3f}" if uni_auc[c] >= 0.5
           else f"AUC {1 - uni_auc[c]:.3f} (reversed)")
    ax.set_title(lab, loc="right", fontsize=11, color=INK2)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\hat{P}$(malignant)")
fig.suptitle("Size and shape almost separate the two groups;\n"
             "texture and smoothness only shift the odds",
             x=0.5, y=1.04, fontsize=15)
fig.text(0.5, -0.03, "Black curve = kernel-smoothed $\\hat{P}$(malignant | x), no "
         "model shape assumed.  Red dots = observed rate in eighths of the data.\n"
         "Dashed line = the 37% base rate.  Ticks = individual masses "
         "(bottom: benign, top: malignant).",
         ha="center", fontsize=10.5, color=INK2)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig2_curves.png"); plt.close(fig)

# ============ 7. TWO FEATURES AT ONCE: THE DECISION SURFACE ================
# %%
# Best 2-feature pair, restricted to low-correlation pairs. This loop is a
# SCREEN, not an estimate: it maximises CV AUC over ~190 candidate pairs, so
# the winning value is optimistically biased (winner's curse) and is not
# quoted anywhere. The reported number is recomputed below for the one pair
# this picks, using the same model the figure draws.
# Pool must reach past the size/shape block -- texture ranks 17th on its own
# but is the natural complement to size, and that is the point of the figure.
cands = list(uni_pow.head(20).index)
best_pair, best_pair_auc = None, -1
for i in range(len(cands)):
    for j in range(i + 1, len(cands)):
        a, b = cands[i], cands[j]
        if abs(corr.loc[a, b]) >= 0.50:      # insist the two be complementary
            continue
        s = []
        for tr_i, va_i in cv.split(Xtr, ytr):
            p = pipe(LogisticRegression(max_iter=5000))
            p.fit(Xtr.iloc[tr_i][[a, b]], ytr[tr_i])
            s.append(roc_auc_score(ytr[va_i],
                                   p.predict_proba(Xtr.iloc[va_i][[a, b]])[:, 1]))
        if np.mean(s) > best_pair_auc:
            best_pair_auc, best_pair = np.mean(s), (a, b)
A, Bf = best_pair
# The figure below draws an RBF-SVM surface. Quote that model's CV AUC, not the
# logistic screen's, or the number in the title describes a different model
# from the one in the picture.
pair_cv = []
for tr_i, va_i in cv.split(Xtr, ytr):
    q = pipe(SVC(kernel="rbf", probability=True, C=1, random_state=SEED))
    q.fit(Xtr.iloc[tr_i][[A, Bf]], ytr[tr_i])
    pair_cv.append(roc_auc_score(
        ytr[va_i], q.predict_proba(Xtr.iloc[va_i][[A, Bf]])[:, 1]))
OUT["pair"] = [pretty(A), pretty(Bf)]
OUT["pair_auc"] = round(float(np.mean(pair_cv)), 4)
OUT["pair_screen_auc"] = round(float(best_pair_auc), 4)
OUT["pair_r"] = round(float(corr.loc[A, Bf]), 2)
print(f"\nBest low-correlation pair: {OUT['pair']} -> RBF-SVM CV AUC "
      f"{OUT['pair_auc']} (r={OUT['pair_r']}; logistic screen said "
      f"{OUT['pair_screen_auc']}, biased high by the search)")

surf = pipe(SVC(kernel="rbf", probability=True, C=1, random_state=SEED))
surf.fit(Xtr[[A, Bf]], ytr)
ax_lo = [np.percentile(Xtr[c], 0.5) for c in (A, Bf)]
ax_hi = [np.percentile(Xtr[c], 99.5) for c in (A, Bf)]
gx = np.linspace(ax_lo[0], ax_hi[0], 300)
gy = np.linspace(ax_lo[1], ax_hi[1], 300)
GX, GY = np.meshgrid(gx, gy)
# Grid as a frame, not a bare array: surf was fit with feature names, and
# predicting on a nameless array warns.
grid_df = pd.DataFrame(np.c_[GX.ravel(), GY.ravel()], columns=[A, Bf])
Z = surf.predict_proba(grid_df)[:, 1].reshape(GX.shape)

fig, ax = plt.subplots(figsize=(8.6, 6.6))
im = ax.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap=DIVERGE, alpha=.92)
ax.contour(GX, GY, Z, levels=[0.5], colors=[INK], linewidths=2.2)
for lab, cl, mk in [(0, BENIGN, "o"), (1, MALIG, "^")]:
    s = ytr == lab
    ax.plot(Xtr[A][s], Xtr[Bf][s], mk, color=cl, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=0.7, linestyle="none",
            alpha=.95, label=["Benign", "Malignant"][lab])
ax.set_xlim(ax_lo[0], ax_hi[0]); ax.set_ylim(ax_lo[1], ax_hi[1])
ax.set_xlabel(pretty(A)); ax.set_ylabel(pretty(Bf))
ax.grid(alpha=.25)
ax.legend(loc="lower right", facecolor=SURFACE, framealpha=.9, frameon=True,
          edgecolor=GRIDC)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                  ticks=[0, .25, .5, .75, 1])
cb.set_label(r"$\hat{P}$(malignant)", color=INK2); cb.outline.set_visible(False)
ax.set_title(f"{pretty(A).capitalize()} and {pretty(Bf)}\n"
             f"(cross-validated AUC {OUT['pair_auc']:.3f}, r = {OUT['pair_r']})",
             loc="left", pad=12)
ax.text(0, -0.155, "Black line = the 50% decision boundary. Shading and AUC "
        "are both from the RBF-kernel SVM drawn here. Best pair of a search, "
        "so the AUC is a\nceiling rather than an unbiased estimate.",
        transform=ax.transAxes, fontsize=10.5, color=INK2)
fig.savefig(f"{FIGDIR}/fig3_surface.png"); plt.close(fig)

# ===================== 8. CHOOSE THE OPERATING POINT, THEN TEST ONCE =======
# %%
# The metric that matters: a false negative sends a woman with cancer home.
# Pick the threshold on TRAINING out-of-fold predictions to hit >=98% sensitivity.
TARGET_SENS = 0.98
final = fitted[BEST]
oof = cross_val_predict(final, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
fpr_o, tpr_o, thr_o = roc_curve(ytr, oof)
ok = tpr_o >= TARGET_SENS
# Among thresholds meeting the sensitivity floor, take the one with the fewest
# false alarms (equivalently the highest threshold, since ROC is monotone).
THRESH = float(thr_o[ok][np.argmin(fpr_o[ok])])
OUT["threshold"] = round(THRESH, 3)
OUT["target_sens"] = int(TARGET_SENS * 100)
oof_sens = ((oof >= THRESH) & (ytr == 1)).sum() / (ytr == 1).sum()
oof_spec = ((oof < THRESH) & (ytr == 0)).sum() / (ytr == 0).sum()
OUT["oof_sens"] = round(100 * oof_sens, 1)
OUT["oof_spec"] = round(100 * oof_spec, 1)
print(f"\nThreshold {OUT['threshold']} chosen on training out-of-fold predictions "
      f"(sens {OUT['oof_sens']}%, spec {OUT['oof_spec']}%).")

# ---- reduced model: the same thing on K named features ----
reduced = pipe(LogisticRegression(max_iter=5000, C=1))
reduced.fit(Xtr[FINAL_FEATURES], ytr)
oof_r = cross_val_predict(reduced, Xtr[FINAL_FEATURES], ytr, cv=cv,
                          method="predict_proba")[:, 1]
f2, t2, th2 = roc_curve(ytr, oof_r)
ok2 = t2 >= TARGET_SENS
THRESH_R = float(th2[ok2][np.argmax(1 - f2[ok2])])
OUT["threshold_reduced"] = round(THRESH_R, 3)

# ================================================================ TEST =====
# %%
# Everything above used the training set only. This block runs once, and scores
# the two models that were fully specified before it ran -- the full-feature
# preferred model and the K-feature reduced one. Nothing below re-tunes anything.
#
# Caveat carried into the report: THRESH was picked on out-of-fold probabilities,
# but is applied to a model refit on all of Xtr. The refit model's probabilities
# are a little sharper, so the 98% sensitivity floor transfers approximately.
final.fit(Xtr, ytr)
pte = final.predict_proba(Xte)[:, 1]
pred = (pte >= THRESH).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
OUT.update(
    test_auc=round(float(roc_auc_score(yte, pte)), 4),
    test_acc=round(100 * accuracy_score(yte, pred), 1),
    test_sens=round(100 * tp / (tp + fn), 1),
    test_spec=round(100 * tn / (tn + fp), 1),
    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
)
reduced.fit(Xtr[FINAL_FEATURES], ytr)
pte_r = reduced.predict_proba(Xte[FINAL_FEATURES])[:, 1]
pred_r = (pte_r >= THRESH_R).astype(int)
tn2, fp2, fn2, tp2 = confusion_matrix(yte, pred_r).ravel()
OUT.update(
    test_auc_reduced=round(float(roc_auc_score(yte, pte_r)), 4),
    test_acc_reduced=round(100 * accuracy_score(yte, pred_r), 1),
    test_sens_reduced=round(100 * tp2 / (tp2 + fn2), 1),
    test_spec_reduced=round(100 * tn2 / (tn2 + fp2), 1),
    fn_reduced=int(fn2), fp_reduced=int(fp2),
)
# per-100 rates, because counts on 114 cases are not the useful unit
OUT["fn_per100_malig"] = round(100 * fn / (tp + fn), 1)
OUT["fp_per100_benign"] = round(100 * fp / (tn + fp), 1)
# overfitting check
OUT["train_auc_best"] = round(float(roc_auc_score(ytr, final.predict_proba(Xtr)[:, 1])), 4)

print(f"\n=== TEST SET (one pass, two pre-specified models, n={OUT['n_test']}) ===")
print(f"{BEST}: AUC {OUT['test_auc']}, accuracy {OUT['test_acc']}% "
      f"(baseline {OUT['baseline_acc']}%), sensitivity {OUT['test_sens']}%, "
      f"specificity {OUT['test_spec']}%  [FN={fn}, FP={fp}]")
print(f"{K} named features: AUC {OUT['test_auc_reduced']}, "
      f"accuracy {OUT['test_acc_reduced']}%, sens {OUT['test_sens_reduced']}%, "
      f"spec {OUT['test_spec_reduced']}%  [FN={fn2}, FP={fp2}]")

# Fig 7: test ROC + confusion at the chosen operating point.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
f, t, _ = roc_curve(yte, pte)
fr, tr_, _ = roc_curve(yte, pte_r)
a1.plot([0, 1], [0, 1], color=GRIDC, lw=1.5, ls=(0, (4, 3)))
a1.plot(f, t, color=BENIGN, lw=2.4, label=f"All 30 (AUC {OUT['test_auc']:.3f})")
a1.plot(fr, tr_, color=MALIG, lw=2.4, ls=(0, (5, 2)),
        label=f"{K} named (AUC {OUT['test_auc_reduced']:.3f})")
a1.plot(1 - OUT["test_spec"] / 100, OUT["test_sens"] / 100, "o", color=INK,
        markersize=10, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
a1.annotate("operating point", xy=(1 - OUT["test_spec"] / 100, OUT["test_sens"] / 100),
            xytext=(0.34, 0.55), fontsize=11, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
a1.set_xlabel("1 − specificity"); a1.set_ylabel("Sensitivity")
a1.set_xlim(-0.02, 1.02); a1.set_ylim(-0.02, 1.02)
a1.legend(loc="lower right"); a1.set_title("Test-set ROC", loc="left")

cm = np.array([[tn, fp], [fn, tp]])
a2.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=BLUE_SEQ, vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        a2.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=22,
                color=SURFACE if cm[i, j] / cm[i].sum() > .5 else INK)
a2.set_xticks([0, 1]); a2.set_xticklabels(["called benign", "called malignant"])
a2.set_yticks([0, 1]); a2.set_yticklabels(["truly\nbenign", "truly\nmalignant"])
a2.grid(False); a2.set_title(f"At threshold {OUT['threshold']:.2f}", loc="left")
fig.text(0.5, -0.04, f"Shading is row-wise rate. {OUT['test_sens']}% of "
         f"malignant masses caught; {OUT['fp_per100_benign']} of every 100 benign "
         "masses flagged for follow-up.",
         ha="center", fontsize=10.5, color=INK2)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig7_test.png"); plt.close(fig)

# %%
with open("results.json", "w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print("\nWrote results.json and 7 figures to ./figures/")
