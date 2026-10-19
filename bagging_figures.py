"""
Bagging / Random Forest visualization for ML 101.
Generates a numbered sequence of figures saved to ./bagging_figures/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUT = "bagging_figures"
os.makedirs(OUT, exist_ok=True)

RNG = np.random.RandomState(42)
PALETTE = ["#4C72B0", "#DD8452"]
CMAP_BG = ListedColormap(["#c7d9f0", "#f5d5b8"])
CMAP_HARD = ListedColormap(["#4C72B0", "#DD8452"])

# ── shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
N = 300
X, y = make_moons(n_samples=N, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

def decision_grid(X, pad=0.4, h=0.01):
    x0 = np.arange(X[:, 0].min() - pad, X[:, 0].max() + pad, h)
    x1 = np.arange(X[:, 1].min() - pad, X[:, 1].max() + pad, h)
    xx, yy = np.meshgrid(x0, x1)
    return xx, yy

def plot_boundary(ax, clf, X, xx, yy, title, alpha_bg=0.5):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=alpha_bg, cmap=CMAP_BG)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(*X[y_train == 0].T, c=PALETTE[0], s=22, edgecolors="w", lw=0.4, zorder=3)
    ax.scatter(*X[y_train == 1].T, c=PALETTE[1], s=22, edgecolors="w", lw=0.4, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — The dataset
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 1: dataset …")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Fig 1 — The Dataset", fontsize=14, fontweight="bold", y=1.01)

for ax, (Xp, yp, tag) in zip(axes,
        [(X_train, y_train, "Training set"), (X_test, y_test, "Test set")]):
    for c, col in enumerate(PALETTE):
        mask = yp == c
        ax.scatter(*Xp[mask].T, c=col, s=30, edgecolors="w", lw=0.5,
                   label=f"Class {c}", zorder=3)
    ax.set_title(f"{tag}  (n={len(yp)})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.legend(frameon=False, fontsize=9)

axes[0].set_title(f"Training set  (n={len(y_train)})", fontsize=11, fontweight="bold")
axes[1].set_title(f"Test set  (n={len(y_test)})", fontsize=11, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig01_dataset.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Bootstrap samples (3 of them)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 2: bootstrap samples …")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Fig 2 — Bootstrap Samples  (sampling with replacement)",
             fontsize=13, fontweight="bold", y=1.01)

n_train = len(X_train)
for k, ax in enumerate(axes):
    idx = RNG.choice(n_train, size=n_train, replace=True)
    oob_mask = np.ones(n_train, dtype=bool)
    oob_mask[np.unique(idx)] = False

    Xb, yb = X_train[idx], y_train[idx]
    for c, col in enumerate(PALETTE):
        m = yb == c
        ax.scatter(*Xb[m].T, c=col, s=20, alpha=0.7, edgecolors="w", lw=0.3, label=f"Class {c}")
    # highlight out-of-bag
    ax.scatter(*X_train[oob_mask].T, marker="x", c="gray", s=25, lw=1.0,
               alpha=0.6, label="Out-of-bag")
    oob_frac = oob_mask.sum() / n_train
    ax.set_title(f"Bootstrap sample {k+1}\n(~{oob_frac:.0%} OOB)", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.legend(frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
savefig(fig, "fig02_bootstrap_samples.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — One deep tree: training fit vs test
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3: single deep tree …")
tree = DecisionTreeClassifier(max_depth=None, random_state=0)
tree.fit(X_train, y_train)
tr_acc = accuracy_score(y_train, tree.predict(X_train))
te_acc = accuracy_score(y_test, tree.predict(X_test))

xx, yy = decision_grid(X)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Fig 3 — A Single Decision Tree (max depth = ∞)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, (Xp, yp, tag, acc) in zip(axes, [
        (X_train, y_train, f"Training set  acc={tr_acc:.2f}", tr_acc),
        (X_test,  y_test,  f"Test set  acc={te_acc:.2f}",    te_acc)]):
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.45, cmap=CMAP_BG)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.0)
    for c, col in enumerate(PALETTE):
        m = yp == c
        ax.scatter(*Xp[m].T, c=col, s=25, edgecolors="w", lw=0.4, zorder=3)
    ax.set_title(tag, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
savefig(fig, "fig03_single_tree.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Six individual trees (each trained on a bootstrap sample)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 4: six individual trees …")
N_TREES_SHOW = 6
individual_trees = []
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("Fig 4 — Six Individual Trees  (each trained on one bootstrap sample)\n"
             "Each tree memorises its sample — decision boundaries are jagged and inconsistent",
             fontsize=12, fontweight="bold", y=1.01)

n_train = len(X_train)
for k, ax in enumerate(axes.flat):
    idx = RNG.choice(n_train, size=n_train, replace=True)
    t = DecisionTreeClassifier(max_depth=None, random_state=k)
    t.fit(X_train[idx], y_train[idx])
    individual_trees.append(t)

    tr_a = accuracy_score(y_train, t.predict(X_train))
    te_a = accuracy_score(y_test,  t.predict(X_test))

    Z = t.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.45, cmap=CMAP_BG)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=0.9)
    for c, col in enumerate(PALETTE):
        m = y_train == c
        ax.scatter(*X_train[m].T, c=col, s=14, edgecolors="w", lw=0.3, zorder=3)
    ax.set_title(f"Tree {k+1}   train={tr_a:.2f}  test={te_a:.2f}", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
savefig(fig, "fig04_individual_trees.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Majority-vote boundary (our 6 trees) vs test data
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 5: ensemble boundary …")

# Majority vote from the 6 trees
votes = np.array([t.predict(np.c_[xx.ravel(), yy.ravel()]) for t in individual_trees])
Z_bag = (votes.mean(axis=0) >= 0.5).astype(int).reshape(xx.shape)

# Proper bagging with 100 trees for accuracy numbers
bag100 = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100, random_state=42
)
bag100.fit(X_train, y_train)
Z_bag100 = bag100.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
bag_tr = accuracy_score(y_train, bag100.predict(X_train))
bag_te = accuracy_score(y_test,  bag100.predict(X_test))
tree_te = accuracy_score(y_test, tree.predict(X_test))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fig 5 — Majority Vote Smooths the Boundary",
             fontsize=13, fontweight="bold", y=1.01)

# left: 6-tree vote
ax = axes[0]
ax.contourf(xx, yy, Z_bag, alpha=0.45, cmap=CMAP_BG)
ax.contour(xx, yy, Z_bag, levels=[0.5], colors="k", linewidths=1.5)
for c, col in enumerate(PALETTE):
    m = y_test == c
    ax.scatter(*X_test[m].T, c=col, s=28, edgecolors="w", lw=0.4, zorder=3)
ax.set_title(f"Majority vote of 6 trees\n(test data shown)", fontsize=11, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])

# right: 100-tree ensemble
ax = axes[1]
ax.contourf(xx, yy, Z_bag100, alpha=0.45, cmap=CMAP_BG)
ax.contour(xx, yy, Z_bag100, levels=[0.5], colors="k", linewidths=1.5)
for c, col in enumerate(PALETTE):
    m = y_test == c
    ax.scatter(*X_test[m].T, c=col, s=28, edgecolors="w", lw=0.4, zorder=3)
ax.set_title(
    f"100-tree bagging ensemble\ntrain={bag_tr:.2f}  test={bag_te:.2f}  "
    f"(single tree test={tree_te:.2f})",
    fontsize=11, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
savefig(fig, "fig05_ensemble_boundary.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Accuracy vs number of trees (bias–variance story)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 6: accuracy vs n_trees …")
ns = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 500, 1000, 2000, 5000, 10000]
tr_accs, te_accs = [], []

for n in ns:
    clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=None),
        n_estimators=n, random_state=42
    )
    clf.fit(X_train, y_train)
    tr_accs.append(accuracy_score(y_train, clf.predict(X_train)))
    te_accs.append(accuracy_score(y_test,  clf.predict(X_test)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(tree_te, color="gray", lw=1.2, ls="--", label=f"Single deep tree (test={tree_te:.2f})")
ax.plot(ns, tr_accs, "o-", color=PALETTE[0], lw=2, ms=5, label="Bagging — train")
ax.plot(ns, te_accs, "s-", color=PALETTE[1], lw=2, ms=5, label="Bagging — test")
ax.set_xlabel("Number of trees", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Fig 6 — More Trees → Lower Variance, Better Generalization",
             fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=10)
ax.set_ylim(0.70, 1.02)
ax.set_xscale("log")

# annotation
ax.annotate("Individual trees overfit\n(train acc = 1.0, high variance)",
            xy=(1, 1.0), xytext=(3, 0.96),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")
ax.annotate("Ensemble variance\nreduces quickly",
            xy=(10, te_accs[5]), xytext=(20, 0.83),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")

fig.tight_layout()
savefig(fig, "fig06_accuracy_vs_ntrees.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Soft probability map (show how averaging gives calibrated probs)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 7: probability maps …")

def prob_map(clf, xx, yy):
    return clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Fig 7 — Hard Votes vs Soft Probability:  Averaging Gives Confidence",
             fontsize=12, fontweight="bold", y=1.01)

for ax, (clf, title) in zip(axes, [
    (tree,   "Single tree\n(hard 0/1 everywhere)"),
    (bag100, "100-tree bagging\n(soft probability ≈ confidence)"),
    (BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=None),
                       n_estimators=5, random_state=42).fit(X_train, y_train),
     "5-tree bagging\n(probability, medium smoothing)"),
]):
    P = prob_map(clf, xx, yy)
    im = ax.contourf(xx, yy, P, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(xx, yy, P, levels=[0.5], colors="k", linewidths=1.5)
    for c, col in enumerate(PALETTE):
        m = y_test == c
        ax.scatter(*X_test[m].T, c=col, s=22, edgecolors="w", lw=0.4, zorder=3)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

fig.tight_layout()
savefig(fig, "fig07_probability_maps.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — MNIST samples + depth-1 stump accuracy vs bagging accuracy
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 8: MNIST panel …")
mnist = fetch_openml("mnist_784", version=1)
Xm = mnist.data.values if hasattr(mnist.data, "values") else mnist.data
Xm = Xm.astype(float) / 255.0
ym = mnist.target.astype(int).values if hasattr(mnist.target, "values") else mnist.target.astype(int)

# quick subset for speed
N_MNIST = 5000
idx = RNG.choice(len(Xm), N_MNIST, replace=False)
Xm, ym = Xm[idx], ym[idx]
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(
    Xm, ym, test_size=0.3, random_state=42, stratify=ym
)

# single stump
stump = DecisionTreeClassifier(max_depth=1)
stump.fit(Xm_tr, ym_tr)
acc_stump = accuracy_score(ym_te, stump.predict(Xm_te))

# bagging of stumps
bag_stumps = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200, random_state=42, n_jobs=-1
)
bag_stumps.fit(Xm_tr, ym_tr)
acc_bag_stumps = accuracy_score(ym_te, bag_stumps.predict(Xm_te))

# deep tree
deep = DecisionTreeClassifier(max_depth=None)
deep.fit(Xm_tr, ym_tr)
acc_deep = accuracy_score(ym_te, deep.predict(Xm_te))

# bagging of deep trees
bag_deep = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=50, random_state=42, n_jobs=-1
)
bag_deep.fit(Xm_tr, ym_tr)
acc_bag_deep = accuracy_score(ym_te, bag_deep.predict(Xm_te))

fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.3, wspace=0.1)

# top row: sample digits
for d in range(10):
    ax = fig.add_subplot(gs[0, d])
    digit_idx = np.where(ym_tr == d)[0][0]
    ax.imshow(Xm_tr[digit_idx].reshape(28, 28), cmap="gray_r")
    ax.set_title(str(d), fontsize=10)
    ax.axis("off")

# bottom row: bar chart spanning all columns
ax_bar = fig.add_subplot(gs[1, :])
labels = ["Single\nstump\n(depth 1)", "Bagging\n200 stumps", "Single deep\ntree", "Bagging\n50 deep trees"]
accs   = [acc_stump, acc_bag_stumps, acc_deep, acc_bag_deep]
colors = [PALETTE[0], PALETTE[1], PALETTE[0], PALETTE[1]]
bars = ax_bar.bar(labels, accs, color=colors, edgecolor="w", width=0.5)
ax_bar.set_ylim(0, 1.08)
ax_bar.set_ylabel("Test accuracy (MNIST, 5 k subset)", fontsize=10)
for bar, acc in zip(bars, accs):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.set_title("Bagging lifts accuracy — on MNIST too", fontsize=11, fontweight="bold")

fig.suptitle("Fig 8 — Bagging on MNIST (10-class digit recognition)",
             fontsize=13, fontweight="bold", y=1.01)
savefig(fig, "fig08_mnist.png")

print("\nDone! All figures saved to:", OUT)
print("\nPresentation sequence:")
for i, desc in enumerate([
    "fig01_dataset.png          — introduce the problem (2-class moons)",
    "fig02_bootstrap_samples.png — what bootstrap sampling looks like",
    "fig03_single_tree.png      — one deep tree: perfect train, bad test",
    "fig04_individual_trees.png — 6 trees: all overfit, all different",
    "fig05_ensemble_boundary.png— majority vote smooths the boundary",
    "fig06_accuracy_vs_ntrees.png— variance falls, test accuracy rises",
    "fig07_probability_maps.png — averaging ≈ calibrated probability",
    "fig08_mnist.png            — same story on real data (MNIST)",
], 1):
    print(f"  {i}. {desc}")
