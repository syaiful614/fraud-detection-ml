"""Matplotlib figures used in the README and reports (headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlab.data import V_COLUMNS  # noqa: E402
from fraudlab.metrics import CostCurve  # noqa: E402

LEGIT, FRAUD, GREY, ACCENT = "#3b78b5", "#d1403a", "#8a8f98", "#e08a1e"
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "axes.titleweight": "bold",
    }
)


def _save(fig: plt.Figure, path: Path, tight: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eda(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    hours = (df["Time"] // 7200).astype(int)
    counts = df.groupby(hours)["Class"].agg(["sum", "size"])
    axes[0].bar(counts.index * 2, counts["sum"], width=1.8, align="edge", color=FRAUD)
    axes[0].set_title("Fraud cases per 2-hour window")
    axes[0].set_xlabel("Hours since first transaction")
    axes[0].set_ylabel("Fraud cases")

    bins = np.linspace(0, np.log10(1 + df["Amount"].max()), 40)
    for label, color, name in ((0, LEGIT, "Legitimate"), (1, FRAUD, "Fraud")):
        values = np.log10(1 + df.loc[df["Class"] == label, "Amount"])
        axes[1].hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.5,
            color=color,
            label=name,
        )
    ticks = [0, 1, 2, 3, 4]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([f"{10**t - 1:,.0f}" for t in ticks])
    axes[1].set_title("Transaction amount (log scale, density)")
    axes[1].set_xlabel("Amount")
    axes[1].legend()

    corr = df[V_COLUMNS].corrwith(df["Class"]).sort_values(key=np.abs, ascending=False).head(10)
    axes[2].barh(
        corr.index[::-1],
        corr.values[::-1],
        color=[FRAUD if v > 0 else LEGIT for v in corr.values[::-1]],
    )
    axes[2].set_title("Top 10 |correlation| with fraud")
    axes[2].set_xlabel("Pearson correlation with Class")
    return _save(fig, path)


def plot_validation_scheme(n_dev: int, n_test: int, splits, y_dev: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    total = n_dev + n_test
    for i, (tr, va) in enumerate(splits):
        y = len(splits) - i
        ax.barh(y, tr[-1] - tr[0] + 1, left=tr[0], color=LEGIT, height=0.55)
        ax.barh(y, va[-1] - va[0] + 1, left=va[0], color=ACCENT, height=0.55)
        ax.text(
            va[-1] + total * 0.005,
            y,
            f"{int(y_dev.iloc[va].sum())} frauds",
            va="center",
            fontsize=8,
            color="#444",
        )
        ax.text(-total * 0.01, y, f"CV fold {i + 1}", ha="right", va="center", fontsize=9)
    ax.barh(0, n_dev, left=0, color=LEGIT, height=0.55)
    ax.barh(0, n_test, left=n_dev, color=FRAUD, height=0.55)
    ax.text(-total * 0.01, 0, "Final fit → test", ha="right", va="center", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, total * 1.1)
    ax.set_xlabel("Transactions in chronological order →")
    ax.set_title("Validation scheme: expanding-window CV on the past, untouched future test set")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LEGIT, ACCENT, FRAUD)]
    ax.legend(handles, ["train", "validation", "test (used once)"], loc="upper right")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    return _save(fig, path)


def plot_cv_comparison(summary: list[dict], prevalence: float, path: Path) -> Path:
    df = pd.DataFrame(summary).sort_values("ap_mean")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 1.6))
    colors = [GREY if m in ("dummy", "isolation_forest") else LEGIT for m in df["model"]]
    ax.barh(
        df["model"],
        df["ap_mean"],
        xerr=df["ap_std"],
        color=colors,
        error_kw={"ecolor": "#333", "capsize": 3},
    )
    for y, (v, sd) in enumerate(zip(df["ap_mean"], df["ap_std"], strict=True)):
        ax.text(v + sd + 0.012, y, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(prevalence, color=FRAUD, ls="--", lw=1)
    ax.text(prevalence + 0.005, -0.45, "random = prevalence", color=FRAUD, fontsize=8)
    ax.set_xlim(0, min(1.0, df["ap_mean"].max() + 0.15))
    ax.set_xlabel("Average precision (mean ± std over time-ordered folds)")
    ax.set_title("Model comparison (cross-validation on development period)")
    return _save(fig, path)


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    prevalence: float,
    operating_point: tuple[float, float],
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 5))
    for name, (prec, rec, ap) in curves.items():
        main = name == "final (tuned LightGBM)"
        ax.plot(
            rec,
            prec,
            lw=2.6 if main else 1.4,
            color=FRAUD if main else None,
            ls="-" if main else "--",
            label=f"{name}  (AP={ap:.3f})",
        )
    ax.axhline(prevalence, color=GREY, ls=":", lw=1)
    ax.scatter(
        *operating_point, marker="*", s=220, color="black", zorder=5, label="chosen operating point"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall on the held-out test period")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def plot_cost_curve(
    curve: CostCurve, n_rows: int, best: dict, path: Path, max_alert_rate: float = 0.01
) -> Path:
    kmax = int(max_alert_rate * n_rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(100 * curve.alerts[:kmax] / n_rows, curve.cost[:kmax], color=LEGIT, lw=2)
    ax.axhline(best["baseline_cost"], color=GREY, ls="--", lw=1)
    ax.text(
        0.003,
        best["baseline_cost"] * 0.97,
        "no model (all fraud slips through)",
        color=GREY,
        fontsize=8,
        va="top",
    )
    ax.scatter(
        100 * best["alerts"] / n_rows,
        best["cost"],
        color=FRAUD,
        s=70,
        zorder=5,
        label=f"cost-optimal: {best['alerts']} alerts",
    )
    ax.set_xlabel("Share of transactions flagged for review (%)")
    ax.set_ylabel("Total cost (missed fraud + review effort)")
    ax.set_title("Choosing the decision threshold by money, not by 0.5")
    ax.legend()
    return _save(fig, path)


def plot_score_distribution(y: np.ndarray, s: np.ndarray, threshold: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(s[y == 0], bins=bins, color=LEGIT, alpha=0.6, label="Legitimate")
    ax.hist(s[y == 1], bins=bins, color=FRAUD, alpha=0.7, label="Fraud")
    ax.axvline(threshold, color="black", ls="--", lw=1.2, label=f"threshold = {threshold:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Score distribution on the test period")
    ax.legend()
    return _save(fig, path)


def plot_shap_summary(
    shap_values: np.ndarray, features: pd.DataFrame, path: Path, top: int = 12, seed: int = 0
) -> Path:
    rng = np.random.default_rng(seed)
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1][:top]
    names = features.columns.to_numpy()[order]

    fig, (ax_bar, ax_dot) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.45})
    ax_bar.barh(names[::-1], importance[order][::-1], color=LEGIT)
    ax_bar.set_title("Global importance (mean |SHAP|)")
    ax_bar.set_xlabel("mean |SHAP value| (log-odds)")

    sc = None
    for rank, col in enumerate(order[::-1]):
        vals = features.iloc[:, col].to_numpy()
        pct = pd.Series(vals).rank(pct=True).to_numpy()
        y = rank + rng.normal(0, 0.11, len(vals)).clip(-0.35, 0.35)
        sc = ax_dot.scatter(
            shap_values[:, col], y, c=pct, cmap="coolwarm", s=6, alpha=0.65, linewidths=0
        )
    ax_dot.set_yticks(range(len(order)))
    ax_dot.set_yticklabels(names[::-1])
    ax_dot.axvline(0, color="#444", lw=0.8)
    ax_dot.set_xlabel("SHAP value (impact on log-odds of fraud)")
    ax_dot.set_title("How feature values push the prediction")
    ax_dot.grid(False, axis="y")
    cbar = fig.colorbar(sc, ax=ax_dot, pad=0.02)
    cbar.set_label("feature value (percentile)")
    return _save(fig, path, tight=False)


def plot_drift(drift: pd.DataFrame, path: Path, top: int = 15) -> Path:
    d = drift.head(top).iloc[::-1]
    colors = [FRAUD if s == "major" else ACCENT if s == "moderate" else LEGIT for s in d["status"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(d) + 1.6))
    ax.barh(d["feature"], d["psi"], color=colors)
    ax.axvline(0.1, color=ACCENT, ls="--", lw=1)
    ax.axvline(0.25, color=FRAUD, ls="--", lw=1)
    ax.set_xlabel("PSI: development period vs test period")
    ax.set_title("Feature drift (PSI), top 15")
    return _save(fig, path)
