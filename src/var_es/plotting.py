"""Static PNG figures (matplotlib, non-interactive Agg backend).

Every figure is labelled as synthetic when the data are synthetic, uses losses
(positive = loss) with upper-tail VaR on loss charts, shades predefined stress
periods, and is closed after saving. Losses are shown in % of notional; on the
GBP 1,000,000 reporting notional, 1% corresponds to GBP 10,000.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

MODEL_ORDER = ["historical", "normal", "student_t", "monte_carlo"]
MODEL_LABEL = {
    "historical": "Historical simulation",
    "normal": "Parametric normal",
    "student_t": "Parametric Student-t",
    "monte_carlo": "Monte Carlo (MVN)",
}
# fixed categorical order (never cycled)
MODEL_COLOR = {"historical": "#2a78d6", "normal": "#eb6834", "student_t": "#1baf7a", "monte_carlo": "#eda100"}
EXC_COLOR = "#d03b3b"
LOSS_COLOR = "#8a8984"
STRESS_SHADE = "#e1e0d9"
ZONE_COLOR = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b", "unavailable": "#c3c2b7"}
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": True,
        "legend.facecolor": "#fcfcfb",
        "legend.edgecolor": "#e1e0d9",
        "legend.framealpha": 0.85,
        "lines.linewidth": 1.2,
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _stress_spans(dates: pd.DatetimeIndex, is_stress: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans = []
    s = np.asarray(is_stress, dtype=bool)
    i = 0
    while i < s.size:
        if s[i]:
            j = i
            while j + 1 < s.size and s[j + 1]:
                j += 1
            spans.append((dates[i], dates[j]))
            i = j + 1
        else:
            i += 1
    return spans


def _shade(ax, spans, label: bool = True) -> None:
    for k, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color=STRESS_SHADE, alpha=0.9, lw=0, zorder=0,
                   label="Predefined stress period" if (label and k == 0) else None)


def _date_axis(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _tag(synthetic: bool) -> str:
    return " (synthetic data)" if synthetic else ""


# ---------------------------------------------------------------------- data plots
def plot_portfolio_returns(port: pd.DataFrame, oos_start: pd.Timestamp, path: Path, synthetic: bool = True) -> Path:
    """Daily portfolio simple returns with stress shading and the first forecast date."""
    d = pd.DatetimeIndex(port["date"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _shade(ax, _stress_spans(d, port["is_stress"].to_numpy()))
    ax.plot(d, port["portfolio_return"] * 100, color=MODEL_COLOR["historical"], lw=0.7, label="Portfolio return")
    ax.axvline(oos_start, color=INK2, ls="--", lw=1, label="First out-of-sample forecast")
    ax.set_title("Daily portfolio simple return" + _tag(synthetic))
    ax.set_ylabel("Return (% of portfolio value)")
    ax.set_xlabel("Date (synthetic business days)" if synthetic else "Date")
    _date_axis(ax)
    ax.legend(loc="upper left", ncol=3)
    return _save(fig, path)


def plot_asset_correlations(asset_returns: pd.DataFrame, path: Path, synthetic: bool = True) -> Path:
    """Full-sample sample correlation heatmap (descriptive only; not used for forecasting)."""
    corr = asset_returns.corr().to_numpy()
    names = list(asset_returns.columns)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="#ffffff" if corr[i, j] > 0.6 else INK, fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson correlation of simple returns")
    ax.set_title("Full-sample asset return correlations" + _tag(synthetic) + "\n(descriptive; forecasts use trailing windows only)")
    return _save(fig, path)


def plot_return_distribution(port: pd.DataFrame, path: Path, synthetic: bool = True) -> Path:
    """Histogram with full-sample normal and Student-t fits, plus a normal QQ plot (descriptive)."""
    r = port["portfolio_return"].to_numpy() * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.hist(r, bins=80, density=True, color="#c3c2b7", edgecolor="#fcfcfb", lw=0.3, label="Observed")
    x = np.linspace(r.min(), r.max(), 400)
    a1.plot(x, stats.norm.pdf(x, r.mean(), r.std(ddof=1)), color=MODEL_COLOR["normal"], lw=1.6, label="Normal fit")
    nu, loc, sc = stats.t.fit(r)
    a1.plot(x, stats.t.pdf(x, nu, loc, sc), color=MODEL_COLOR["student_t"], lw=1.6, label=f"Student-t fit (nu={nu:.1f})")
    a1.set_yscale("log")
    a1.set_ylim(bottom=1e-4)
    a1.set_xlabel("Daily portfolio return (%)")
    a1.set_ylabel("Density (log scale)")
    a1.set_title("Return distribution, full sample" + _tag(synthetic))
    a1.legend()
    osm, osr = stats.probplot(r, dist="norm", fit=False)
    a2.plot(osm, osr, "o", ms=2.5, color=MODEL_COLOR["historical"], label="Ordered returns")
    a2.plot(osm, r.mean() + r.std(ddof=1) * osm, color=INK2, lw=1, label="Normal reference")
    a2.set_xlabel("Standard normal quantile")
    a2.set_ylabel("Observed return quantile (%)")
    a2.set_title("Normal QQ plot (descriptive, full sample)")
    a2.legend()
    fig.text(0.01, -0.03, "Full-sample fits are descriptive only; all VaR/ES forecasts are fitted on trailing windows.",
             fontsize=8, color=INK2)
    return _save(fig, path)


# ------------------------------------------------------------------ forecast plots
def _loss_panels(ev: pd.DataFrame, c: float, path: Path, title: str, show_es: bool, synthetic: bool) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        g = ev[(ev["model"] == model) & np.isclose(ev["confidence"], c)].sort_values("forecast_date")
        if g.empty:
            ax.set_visible(False)
            continue
        d = pd.DatetimeIndex(g["forecast_date"])
        _shade(ax, _stress_spans(d, g["is_stress"].to_numpy()))
        ax.plot(d, g["realized_loss"] * 100, color=LOSS_COLOR, lw=0.6, label="Realised loss")
        ax.plot(d, g["var"] * 100, color=MODEL_COLOR[model], lw=1.4, label=f"VaR {c:.1%}")
        if show_es:
            ax.plot(d, g["es"] * 100, color=MODEL_COLOR[model], lw=1.2, ls="--", label=f"ES {c:.1%}")
        hit = g["exception"].fillna(False).to_numpy(dtype=bool)
        n_valid = int(g["forecast_status"].eq("ok").sum())
        ax.scatter(d[hit], g["realized_loss"].to_numpy()[hit] * 100, s=14, color=EXC_COLOR, zorder=3,
                   label=f"VaR exception (n={hit.sum()} of {n_valid})")
        ax.set_title(MODEL_LABEL[model])
        _date_axis(ax)
        ax.legend(loc="upper left", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Loss (% of notional)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Forecast date")
    fig.suptitle(title + _tag(synthetic) + "\nOne-day losses; positive = loss; 1% = GBP 10,000 on GBP 1m notional; shaded = predefined stress",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_var_comparison(ev: pd.DataFrame, c: float, path: Path, synthetic: bool = True) -> Path:
    return _loss_panels(ev, c, path, f"Out-of-sample {c:.0%} one-day VaR vs realised loss", False, synthetic)


def plot_var_es_comparison(ev: pd.DataFrame, c: float, path: Path, synthetic: bool = True) -> Path:
    return _loss_panels(ev, c, path, f"Out-of-sample {c:.1%} one-day VaR and ES vs realised loss", True, synthetic)


def plot_rolling_exception_rates(rates: pd.DataFrame, ev: pd.DataFrame, window: int, path: Path,
                                 synthetic: bool = True) -> Path:
    """Rolling exception rate per confidence (small multiples) with nominal reference rates."""
    levels = sorted(rates["confidence"].unique())
    fig, axes = plt.subplots(len(levels), 1, figsize=(10, 2.6 * len(levels)), sharex=True)
    axes = np.atleast_1d(axes)
    first = ev[ev["model"] == MODEL_ORDER[0]].drop_duplicates("forecast_date").sort_values("forecast_date")
    spans = _stress_spans(pd.DatetimeIndex(first["forecast_date"]), first["is_stress"].to_numpy())
    for ax, c in zip(axes, levels):
        _shade(ax, spans)
        for model in MODEL_ORDER:
            g = rates[(rates["model"] == model) & np.isclose(rates["confidence"], c)].dropna(subset=["forecast_date"])
            ax.plot(pd.DatetimeIndex(g["forecast_date"]), g["rolling_rate"] * 100, color=MODEL_COLOR[model],
                    lw=1.2, label=MODEL_LABEL[model])
        ax.axhline((1 - c) * 100, color=INK, ls=":", lw=1.2, label=f"Nominal {(1 - c):.1%}")
        ax.set_ylabel("Exception rate (%)")
        ax.set_title(f"{c:.1%} VaR: trailing {window}-forecast exception rate")
        _date_axis(ax)
    axes[0].legend(loc="upper left", ncol=3, fontsize=7.5)
    axes[-1].set_xlabel("Window end date")
    fig.suptitle("Rolling VaR exception rates" + _tag(synthetic) + "  (shaded = predefined stress)\n"
                 "Normal and MVN Monte Carlo target the same distribution, so their lines largely overlap", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_stress_detail(ev: pd.DataFrame, c: float, path: Path, pad: int = 20, synthetic: bool = True) -> Path:
    """Zoom on each predefined stress interval: realised loss and VaR for every model."""
    first = ev[(ev["model"] == MODEL_ORDER[0]) & np.isclose(ev["confidence"], c)].sort_values("obs_index")
    spans = _stress_spans(pd.DatetimeIndex(first["forecast_date"]), first["is_stress"].to_numpy())
    if not spans:
        spans = []
    ncol = max(len(spans), 1)
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 3.8), sharey=True, squeeze=False)
    for ax, (a, b) in zip(axes[0], spans):
        pos = first["obs_index"].to_numpy()
        dts = pd.DatetimeIndex(first["forecast_date"])
        ia, ib = int(np.searchsorted(dts, a)), int(np.searchsorted(dts, b))
        lo, hi = dts[max(ia - pad, 0)], dts[min(ib + pad, len(dts) - 1)]
        _ = pos
        ax.axvspan(a, b, color=STRESS_SHADE, lw=0, zorder=0, label="Predefined stress period")
        win = first[(first["forecast_date"] >= lo) & (first["forecast_date"] <= hi)]
        ax.plot(pd.DatetimeIndex(win["forecast_date"]), win["realized_loss"] * 100, color=LOSS_COLOR, lw=0.9,
                marker="o", ms=2.5, label="Realised loss")
        for model in MODEL_ORDER:
            g = ev[(ev["model"] == model) & np.isclose(ev["confidence"], c)]
            g = g[(g["forecast_date"] >= lo) & (g["forecast_date"] <= hi)].sort_values("forecast_date")
            ax.plot(pd.DatetimeIndex(g["forecast_date"]), g["var"] * 100, color=MODEL_COLOR[model], lw=1.4,
                    label=f"{MODEL_LABEL[model]} VaR")
        ax.set_title(f"Stress interval {a.date()} to {b.date()}")
        ax.set_xlabel("Forecast date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
    axes[0][0].set_ylabel("Loss (% of notional)")
    axes[0][0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"{c:.0%} VaR around predefined stress periods" + _tag(synthetic) + f" (+/- {pad} days)", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_traffic_light_history(tl: pd.DataFrame, path: Path, synthetic: bool = True) -> Path:
    """Zone of each rolling 250-day window (99% VaR) per model; educational only."""
    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(10, 1.2 * len(MODEL_ORDER) + 1.4), sharex=True)
    for ax, model in zip(axes, MODEL_ORDER):
        g = tl[tl["model"] == model]
        ax.grid(False)
        if g.empty:
            ax.text(0.5, 0.5, "no complete 250-day window", transform=ax.transAxes, ha="center")
        else:
            d = pd.DatetimeIndex(g["window_end"])
            width = np.r_[np.diff(mdates.date2num(d)), 1.0]
            zone_text = {"green": "green, 0-4", "amber": "amber, 5-9", "red": "red, >=10", "unavailable": "unavailable"}
            for zone, col in ZONE_COLOR.items():
                m = (g["zone"] == zone).to_numpy()
                if not m.any():
                    continue
                ax.bar(mdates.date2num(d)[m], 1, width=width[m], color=col, align="edge", lw=0,
                       label=f"{zone_text[zone]}: {m.sum()} windows")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
        ax.set_yticks([])
        ax.set_ylabel(MODEL_LABEL[model], rotation=0, ha="right", va="center")
        ax.xaxis_date()
        _date_axis(ax)
    axes[-1].set_xlabel("End date of rolling 250-observation window")
    fig.suptitle("Basel-style traffic-light zones for 99% one-day VaR" + _tag(synthetic) +
                 "\nGreen 0-4, amber 5-9, red >=10 exceptions per 250 days; educational illustration only, "
                 "not a regulatory assessment", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_mc_convergence(raw: pd.DataFrame, summary: pd.DataFrame, path: Path, synthetic: bool = True) -> Path:
    """MC VaR/ES estimates across seeds vs analytical normal benchmark, and RMSE vs scenario count."""
    levels = sorted(raw["confidence"].unique())
    c_top = max(levels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    sub = raw[np.isclose(raw["confidence"], c_top)]
    for ax, (col, bench, name) in zip(axes[:2], [("mc_var", "analytic_var", "VaR"), ("mc_es", "analytic_es", "ES")]):
        ns = sorted(sub["n_scenarios"].unique())
        for k, n in enumerate(ns):
            y = sub.loc[sub["n_scenarios"] == n, col] * 100
            ax.scatter(np.full(y.size, k) + np.linspace(-0.15, 0.15, y.size), y, s=12,
                       color=MODEL_COLOR["monte_carlo"], edgecolor="#fcfcfb", lw=0.5,
                       label="MC estimate (one seed)" if k == 0 else None)
        ax.axhline(sub[bench].iloc[0] * 100, color=MODEL_COLOR["normal"], lw=1.4, label="Analytical normal")
        ax.set_xticks(range(len(ns)), [f"{n:,}" for n in ns])
        ax.set_xlabel("Scenarios per forecast")
        ax.set_ylabel(f"{name} (% of notional)")
        ax.set_title(f"{c_top:.0%} {name}: MC across {sub['seed'].nunique()} seeds")
        ax.legend(fontsize=7.5)
    ax = axes[2]
    style = {0.95: "-", 0.975: "--", 0.99: ":"}
    for c in levels:
        g = summary[np.isclose(summary["confidence"], c)].sort_values("n_scenarios")
        ax.plot(g["n_scenarios"], g["var_rmse"] / g["analytic_var"] * 100, marker="o", ms=4,
                color=MODEL_COLOR["historical"], ls=style.get(c, "-"), label=f"VaR {c:.1%}")
        ax.plot(g["n_scenarios"], g["es_rmse"] / g["analytic_es"] * 100, marker="s", ms=4,
                color=MODEL_COLOR["student_t"], ls=style.get(c, "-"), label=f"ES {c:.1%}")
    n = np.array(sorted(summary["n_scenarios"].unique()), dtype=float)
    ref = summary.sort_values("n_scenarios")
    r0 = float((ref["var_rmse"] / ref["analytic_var"]).iloc[0] * 100)
    ax.plot(n, r0 * np.sqrt(n[0] / n), color=INK2, lw=1, ls="-.", label="1/sqrt(N) reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n, [f"{int(v):,}" for v in n])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
    ax.set_xlabel("Scenarios per forecast (log)")
    ax.set_ylabel("RMSE vs analytical (% of benchmark, log)")
    ax.set_title("Relative RMSE across seeds")
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Monte Carlo sampling error at one predefined forecast origin" + _tag(synthetic) +
                 " - MVN Monte Carlo and analytical normal target the same distribution", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)
