#!/usr/bin/env python3
"""
Logistic Regression & SGD — Lecture Visualizations
Navigate: LEFT / RIGHT arrow keys  |  Q to quit
"""

import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # stable on macOS; avoids segfault with default backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── palette ──────────────────────────────────────────────────────────
DARK   = "#0f1117"
PANEL  = "#1a1d27"
BORDER = "#2a2d3a"
TEXT   = "#e0e0e0"
GRAY   = "#666688"
BLUE   = "#4f8ef7"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
ORANGE = "#f39c12"


def ax_style(ax, title="", xlabel="", ylabel="", fontsize=13):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    if title:
        ax.set_title(title, color=TEXT, fontsize=fontsize, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT)


def note(ax, txt, x, y, color=TEXT):
    ax.text(x, y, txt, color=color, fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BORDER, alpha=0.9))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ════════════════════════════════════════════════════════════════════
#  SLIDES
# ════════════════════════════════════════════════════════════════════

SLIDES = []
TITLES = []


def slide(title):
    def decorator(fn):
        SLIDES.append(fn)
        TITLES.append(title)
        return fn
    return decorator


# ── 1 · Sigmoid ──────────────────────────────────────────────────────
@slide("Sigmoid Function")
def s_sigmoid(fig):
    z = np.linspace(-7, 7, 500)
    s = sigmoid(z)
    ds = s * (1 - s)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.plot(z, s, color=BLUE, lw=3)
    ax1.fill_between(z, 0, s, alpha=0.12, color=BLUE)
    ax1.axhline(0.5, color=GRAY, ls="--", lw=1)
    ax1.axvline(0,   color=GRAY, ls="--", lw=1)
    ax1.set_ylim(-0.05, 1.08)
    ax_style(ax1, "σ(z) = 1 / (1 + e⁻ᶻ)", "z", "σ(z)")
    note(ax1, "output ∈ (0, 1)\n→ interpret as probability", 0.35, 0.08)

    ax2.plot(z, ds, color=GREEN, lw=3)
    ax2.fill_between(z, 0, ds, alpha=0.12, color=GREEN)
    ax2.scatter([0], [0.25], color=GREEN, s=90, zorder=5)
    ax2.set_ylim(-0.01, 0.30)
    ax_style(ax2, "σ′(z) = σ(z)(1−σ(z))", "z", "σ′(z)")
    note(ax2, "max gradient = 0.25 at z = 0\ngradient → 0 in saturation zones", 0.18, 0.72)


# ── 2 · Log-Loss vs 0/1 Loss ─────────────────────────────────────────
@slide("Why Log-Loss? Not 0/1 Loss")
def s_loss(fig):
    p = np.linspace(0.001, 0.999, 500)

    ax = fig.add_subplot(111)
    ax.plot(p, -np.log(p),           color=BLUE, lw=3, label="Log-loss  −log(p̂)")
    ax.plot(p, (p < 0.5).astype(float), color=RED,  lw=3, ls="--", label="0/1 loss")
    ax.axvline(0.5, color=GRAY, ls=":", lw=1)
    ax.set_ylim(-0.1, 4.2)
    ax_style(ax, "Loss when true label y = 1", "predicted probability p̂", "loss")
    ax.legend(fontsize=13, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    note(ax, "0/1 loss is flat almost\neverywhere → gradient = 0\n→ SGD cannot learn!", 0.04, 0.65, RED)
    note(ax, "Log-loss is smooth &\ndifferentiable everywhere\n→ clean gradient signal", 0.62, 0.65, BLUE)


# ── 3 · Decision Boundary ────────────────────────────────────────────
@slide("Decision Boundary — 2D Data")
def s_boundary(fig):
    n = 180
    X0 = np.random.randn(n, 2) + [2, 1.5]
    X1 = np.random.randn(n, 2) + [-2, -1.5]
    X  = np.vstack([X0, X1])
    y  = np.hstack([np.ones(n), np.zeros(n)])

    Xb = np.hstack([np.ones((len(X), 1)), X])
    w  = np.zeros(3)
    for _ in range(600):
        p = sigmoid(Xb @ w)
        w -= 0.1 * Xb.T @ (p - y) / len(y)

    xx, yy = np.meshgrid(np.linspace(-6, 6, 250), np.linspace(-6, 6, 250))
    grid   = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()]
    probs  = sigmoid(grid @ w).reshape(xx.shape)

    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, probs, levels=50, cmap="RdBu_r", alpha=0.65)
    ax.contour( xx, yy, probs, levels=[0.5], colors="white", linewidths=2.5)
    ax.scatter(X0[:, 0], X0[:, 1], c=BLUE, s=22, alpha=0.75, label="class 1")
    ax.scatter(X1[:, 0], X1[:, 1], c=RED,  s=22, alpha=0.75, label="class 0")
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax_style(ax, "", "x₁", "x₂")
    ax.legend(fontsize=12, facecolor=PANEL, labelcolor=TEXT)
    note(ax, "white line = decision boundary\nP(y=1|x) = 0.5", 0.55, 0.04)


# ── 4 · Loss Landscape ───────────────────────────────────────────────
@slide("Loss Landscape & Gradient Descent")
def s_landscape(fig):
    np.random.seed(7)
    n  = 80
    x  = np.random.randn(n)
    y  = (x + np.random.randn(n) * 0.5 > 0).astype(float)
    Xb = np.c_[np.ones(n), x]

    def loss_at(w0, w1):
        p = sigmoid(w0 + w1 * x)
        return log_loss(y, p)

    w0s = np.linspace(-3.5, 3.5, 120)
    w1s = np.linspace(-3.5, 5.5, 120)
    W0, W1 = np.meshgrid(w0s, w1s)
    L = np.vectorize(loss_at)(W0, W1)

    # GD trajectory
    w = np.array([-2.8, -2.5])
    traj = [w.copy()]
    for _ in range(60):
        p = sigmoid(Xb @ w)
        w = w - 0.35 * Xb.T @ (p - y) / n
        traj.append(w.copy())
    traj = np.array(traj)
    losses_traj = [loss_at(*t) for t in traj]

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.contourf(W0, W1, L, levels=35, cmap="plasma")
    ax1.contour( W0, W1, L, levels=20, colors="white", alpha=0.15, linewidths=0.5)
    ax1.plot(traj[:, 0], traj[:, 1], "w-o", markersize=4, lw=1.8, alpha=0.9)
    ax1.scatter(traj[0,  0], traj[0,  1], color=RED,   s=110, zorder=6, label="start")
    ax1.scatter(traj[-1, 0], traj[-1, 1], color=GREEN, s=110, zorder=6, label="end")
    ax_style(ax1, "Loss surface (w₀, w₁)", "bias w₀", "weight w₁")
    ax1.legend(fontsize=10, facecolor=PANEL, labelcolor=TEXT)

    ax2.plot(losses_traj, color=BLUE, lw=2.5)
    ax2.scatter(0,             losses_traj[0],  color=RED,   s=80, zorder=5)
    ax2.scatter(len(traj) - 1, losses_traj[-1], color=GREEN, s=80, zorder=5)
    ax_style(ax2, "Loss vs. Step", "gradient descent step", "loss")


# ── 5 · Learning Rates ───────────────────────────────────────────────
@slide("Learning Rate: Too Small / Too Large / Just Right")
def s_lr(fig):
    np.random.seed(3)
    n  = 120
    x  = np.random.randn(n)
    y  = (x + np.random.randn(n) * 0.4 > 0).astype(float)
    Xb = np.c_[np.ones(n), x]

    def run(lr, steps=100):
        w = np.array([-2.5, -2.5])
        hist = []
        for _ in range(steps):
            p = sigmoid(Xb @ w)
            w = w - lr * Xb.T @ (p - y) / n
            hist.append(log_loss(y, sigmoid(Xb @ w)))
        return hist

    ax = fig.add_subplot(111)
    ax.plot(run(0.01), color=GRAY,   lw=2.5, label="too small  (lr = 0.01)")
    ax.plot(run(8.0),  color=RED,    lw=2.5, label="too large  (lr = 8.0)")
    ax.plot(run(0.35), color=GREEN,  lw=2.5, label="just right (lr = 0.35)")
    ax.set_ylim(0, 1.6)
    ax_style(ax, "", "step", "training loss")
    ax.legend(fontsize=13, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    note(ax, "too small → slow convergence", 0.55, 0.82, GRAY)
    note(ax, "too large → diverges / oscillates", 0.55, 0.70, RED)
    note(ax, "just right → fast & stable", 0.55, 0.58, GREEN)


# ── 6 · Batch Size ───────────────────────────────────────────────────
@slide("Batch Size: Full-Batch vs Mini-Batch vs SGD")
def s_batch(fig):
    np.random.seed(5)
    n  = 300
    x  = np.random.randn(n)
    y  = (x + np.random.randn(n) * 0.5 > 0).astype(float)
    Xb = np.c_[np.ones(n), x]

    def run(B, lr, steps=150):
        w = np.array([-2.5, -2.5])
        hist = []
        for _ in range(steps):
            idx = np.random.choice(n, B, replace=False)
            p   = sigmoid(Xb[idx] @ w)
            w   = w - lr * Xb[idx].T @ (p - y[idx]) / B
            hist.append(log_loss(y, sigmoid(Xb @ w)))
        return hist

    ax = fig.add_subplot(111)
    ax.plot(run(n,  0.6), color=GREEN,  lw=2.5, label=f"Full-batch  (B = {n})")
    ax.plot(run(32, 0.3), color=BLUE,   lw=2.0, label="Mini-batch  (B = 32)", alpha=0.9)
    ax.plot(run(1,  0.1), color=ORANGE, lw=1.5, label="SGD         (B = 1)",  alpha=0.8)
    ax.set_ylim(0, 1.2)
    ax_style(ax, "", "step", "loss  (evaluated on full data)")
    ax.legend(fontsize=13, facecolor=PANEL, labelcolor=TEXT)
    note(ax, "Full-batch: smooth but expensive\nMini-batch: best of both worlds\nSGD: noisy but fast & escapes local minima", 0.45, 0.65)


# ── 7 · MNIST Weights ────────────────────────────────────────────────
@slide("MNIST — Learned Weights Resemble Digit Shapes")
def s_mnist(fig):
    # Show a loading message first, then draw
    ax = fig.add_subplot(111)
    ax_style(ax)
    ax.axis("off")
    ax.text(0.5, 0.5, "Loading MNIST…  (first run downloads ~11 MB, then cached)",
            transform=ax.transAxes, ha="center", va="center",
            color=TEXT, fontsize=14)
    fig.canvas.draw()

    try:
        from sklearn.datasets import fetch_openml
        from sklearn.linear_model import LogisticRegression

        mnist  = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X_all, y_all = mnist.data / 255.0, mnist.target.astype(int)

        idx   = np.random.choice(len(X_all), 12000, replace=False)
        X_sub = X_all[idx]; y_sub = y_all[idx]

        clf = LogisticRegression(max_iter=300, C=1.0, solver="saga", multi_class="ovr", n_jobs=-1)
        clf.fit(X_sub, y_sub)

        fig.clf()
        fig.patch.set_facecolor(DARK)
        fig.suptitle("MNIST — Learned Weights Resemble Digit Shapes", color=TEXT, fontsize=16, y=0.98)

        gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.05, wspace=0.05,
                               top=0.88, bottom=0.10, left=0.06, right=0.98)

        for digit in range(10):
            # top row: average images
            ax_avg = fig.add_subplot(gs[0, digit])
            avg    = X_sub[y_sub == digit].mean(axis=0).reshape(28, 28)
            ax_avg.imshow(avg, cmap="gray", interpolation="nearest")
            ax_avg.axis("off")
            ax_avg.set_title(str(digit), color=TEXT, fontsize=12, pad=3)

            # bottom row: weight images
            ax_w  = fig.add_subplot(gs[1, digit])
            wimg  = clf.coef_[digit].reshape(28, 28)
            vmax  = np.abs(wimg).max()
            ax_w.imshow(wimg, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax_w.axis("off")

        fig.text(0.01, 0.73, "Average\nimage",   color=GRAY, fontsize=9, va="center")
        fig.text(0.01, 0.32, "Learned\nweights", color=GRAY, fontsize=9, va="center")
        fig.text(0.5,  0.02,
                 "Red = positive weight (votes for this class)   |   Blue = negative weight",
                 ha="center", color=GRAY, fontsize=10)

    except ImportError:
        fig.clf(); fig.patch.set_facecolor(DARK)
        ax = fig.add_subplot(111); ax_style(ax); ax.axis("off")
        ax.text(0.5, 0.5, "scikit-learn not found.\nRun:  pip install scikit-learn",
                transform=ax.transAxes, ha="center", va="center", color=RED, fontsize=14)
    except Exception as exc:
        fig.clf(); fig.patch.set_facecolor(DARK)
        ax = fig.add_subplot(111); ax_style(ax); ax.axis("off")
        ax.text(0.5, 0.5, f"Error:\n{exc}",
                transform=ax.transAxes, ha="center", va="center", color=RED, fontsize=12)


# ════════════════════════════════════════════════════════════════════
#  NAVIGATION
# ════════════════════════════════════════════════════════════════════

_idx = [0]
_btn_prev = [None]
_btn_next = [None]
_lbl      = [None]


def _draw(fig):
    i = _idx[0]

    # remove old axes but keep the button axes we're about to recreate
    for ax in fig.axes:
        ax.remove()

    fig.patch.set_facecolor(DARK)

    # content area (leaves room at bottom for buttons)
    with plt.rc_context({"axes.facecolor": PANEL}):
        SLIDES[i](fig)

    # slide label
    if _lbl[0] is not None:
        try: _lbl[0].remove()
        except Exception: pass
    _lbl[0] = fig.text(0.5, 0.027, f"Slide {i+1} / {len(SLIDES)}  —  {TITLES[i]}",
                        ha="center", color=GRAY, fontsize=10)

    # ◀ PREV button
    ax_prev = fig.add_axes([0.35, 0.01, 0.12, 0.045])
    ax_prev.set_facecolor(BORDER)
    b_prev = Button(ax_prev, "◀  Prev", color=BORDER, hovercolor="#3a3d4a")
    b_prev.label.set_color(TEXT)
    b_prev.label.set_fontsize(11)
    b_prev.on_clicked(lambda _: _go(-1, fig))
    _btn_prev[0] = b_prev          # keep reference so it isn't GC'd

    # NEXT ▶ button
    ax_next = fig.add_axes([0.53, 0.01, 0.12, 0.045])
    ax_next.set_facecolor(BORDER)
    b_next = Button(ax_next, "Next  ▶", color=BORDER, hovercolor="#3a3d4a")
    b_next.label.set_color(TEXT)
    b_next.label.set_fontsize(11)
    b_next.on_clicked(lambda _: _go(+1, fig))
    _btn_next[0] = b_next

    fig.canvas.draw_idle()


def _go(direction, fig):
    new = _idx[0] + direction
    if 0 <= new < len(SLIDES):
        _idx[0] = new
        _draw(fig)


def _on_key(event, fig):
    if event.key == "right":   _go(+1, fig)
    elif event.key == "left":  _go(-1, fig)
    elif event.key in ("q", "Q", "escape"):
        plt.close("all"); sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            _idx[0] = max(0, min(int(sys.argv[1]) - 1, len(SLIDES) - 1))
        except ValueError:
            pass

    fig = plt.figure(figsize=(14, 7.8))
    fig.canvas.mpl_connect("key_press_event", lambda e: _on_key(e, fig))
    _draw(fig)
    plt.show()
