"""Benchmark visualization: generates comparison plots from results JSON files.

Usage (standalone):
    python -m scripts.plot_results                        # latest results file
    python -m scripts.plot_results --file results/summary_<ts>.json
    python -m scripts.plot_results --out plots/           # custom output dir

Plots produced (saved as PNG + shown interactively when a display is available):
  1. success_rate_bar.png    — horizontal bar chart of success rates
  2. latency_tokens.png      — scatter: latency vs token cost, bubble=quality
  3. radar.png               — radar/spider chart across all 7 metrics
  4. quality_heatmap.png     — heatmap of metric z-scores (relative performance)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Single source of truth — avoids duplicating the mapping in scripts/analyze.py
from scripts.analyze import LEVEL_NAMES


def _load_latest_summary(results_dir: Path) -> list[dict]:
    """Load the most recent summary_*.json file from results_dir."""
    files = sorted(results_dir.glob("summary_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No summary_*.json files found in {results_dir}")
    return json.loads(files[0].read_text())


def _load_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def plot_all(
    summaries: list[dict],
    output_dir: Path,
    show: bool = False,
) -> list[Path]:
    """Generate all benchmark plots.  Returns list of written file paths."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")  # headless mode
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    names = [s["workflow"] for s in summaries]
    n = len(names)
    colors = cm.tab10(np.linspace(0, 0.9, n))  # type: ignore[arg-type]

    # ── 1. Success rate bar ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * n + 1)))
    rates = [s.get("success_rate", 0) * 100 for s in summaries]
    sorted_idx = sorted(range(n), key=lambda i: rates[i])
    sorted_names = [names[i] for i in sorted_idx]
    sorted_rates = [rates[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(sorted_names, sorted_rates, color=sorted_colors, edgecolor="white")
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("Workflow Success Rates", fontweight="bold")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, sorted_rates):
        ax.text(
            val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9,
        )
    ax.axvline(x=np.mean(sorted_rates), color="grey", linestyle="--", linewidth=1, label="mean")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = output_dir / "success_rate_bar.png"
    fig.savefig(p, dpi=150)
    written.append(p)
    if show:
        plt.show()
    plt.close(fig)

    # ── 2. Latency vs Token cost scatter ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (s, c) in enumerate(zip(summaries, colors)):
        latency = s.get("avg_latency_ms", 0)
        tokens = s.get("avg_tokens", 0)
        qs = s.get("avg_quality_score")
        bubble = 200 + (qs or 0.5) * 600
        ax.scatter(latency, tokens, s=bubble, color=c, alpha=0.75, edgecolors="white", linewidths=1)
        ax.annotate(
            s["workflow"], (latency, tokens),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Avg Latency (ms)")
    ax.set_ylabel("Avg Tokens")
    ax.set_title("Latency vs Token Cost\n(bubble size ∝ quality score)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / "latency_tokens.png"
    fig.savefig(p, dpi=150)
    written.append(p)
    if show:
        plt.show()
    plt.close(fig)

    # ── 3. Radar chart ────────────────────────────────────────────────────
    # Labels for inverted metrics clarify that 100% on the axis means
    # "fastest / cheapest / most stable" (i.e. lowest raw value in the run).
    radar_metrics = [
        ("success_rate",        "Success",         True),
        ("tool_accuracy",       "Tool Acc.",        True),
        ("avg_quality_score",   "Quality",          True),
        ("avg_reasoning_steps", "Steps",            True),
        ("avg_latency_ms",      "Speed\n(↑=fast)",  False),  # inverted
        ("avg_tokens",          "Cost\n(↑=cheap)",  False),  # inverted
        ("avg_retries",         "Stability\n(↑=0)", False),  # inverted
    ]
    radar_labels = [lbl for _, lbl, _ in radar_metrics]
    num_axes = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]

    # Normalise each metric to [0,1] across workflows
    raw: dict[str, list[float]] = {}
    for attr, _, higher_is_better in radar_metrics:
        vals = [s.get(attr) for s in summaries]
        vals_clean = [v if v is not None else 0.0 for v in vals]
        lo, hi = min(vals_clean), max(vals_clean)
        span = hi - lo if hi != lo else 1.0
        if higher_is_better:
            raw[attr] = [(v - lo) / span for v in vals_clean]
        else:
            raw[attr] = [(hi - v) / span for v in vals_clean]  # invert: lower = better = higher score

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
    ax.grid(color="grey", alpha=0.3)

    for i, (s, c) in enumerate(zip(summaries, colors)):
        vals = [raw[attr][i] for attr, _, _ in radar_metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=c, linewidth=1.5, label=s["workflow"])
        ax.fill(angles, vals, color=c, alpha=0.08)

    ax.set_title("Multi-Metric Radar Chart", fontweight="bold", pad=20)
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=8, framealpha=0.9,
    )
    fig.text(
        0.5, 0.01,
        "All axes: outer edge = best in run.  "
        "Inverted axes (Speed, Cost, Stability): higher = lower raw value.",
        ha="center", fontsize=7, color="grey",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = output_dir / "radar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    written.append(p)
    if show:
        plt.show()
    plt.close(fig)

    # ── 4. Z-score heatmap ────────────────────────────────────────────────
    heatmap_metrics = [
        ("success_rate",        "Success",   True),
        ("tool_accuracy",       "Tool Acc.", True),
        ("avg_quality_score",   "Quality",   True),
        ("avg_latency_ms",      "Latency",   False),
        ("avg_tokens",          "Tokens",    False),
        ("avg_retries",         "Retries",   False),
        ("avg_reasoning_steps", "Steps",     True),
    ]
    hm_labels = [lbl for _, lbl, _ in heatmap_metrics]
    matrix = np.zeros((n, len(heatmap_metrics)))

    for j, (attr, _, higher_is_better) in enumerate(heatmap_metrics):
        col = np.array([s.get(attr) if s.get(attr) is not None else 0.0 for s in summaries])
        mu, sigma = col.mean(), col.std()
        z = (col - mu) / sigma if sigma > 0 else np.zeros_like(col)
        matrix[:, j] = z if higher_is_better else -z  # always: green = better

    fig, ax = plt.subplots(figsize=(max(8, len(heatmap_metrics) * 0.9), max(4, n * 0.55 + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax.set_xticks(range(len(hm_labels)))
    ax.set_xticklabels(hm_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Relative Performance Heatmap (z-score, green = better)", fontweight="bold")

    for i in range(n):
        for j in range(len(hm_labels)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="z-score")
    fig.tight_layout()
    p = output_dir / "quality_heatmap.png"
    fig.savefig(p, dpi=150)
    written.append(p)
    if show:
        plt.show()
    plt.close(fig)

    return written


def plot_from_raw(
    level_breakdown: dict,
    cost_efficiency: dict,
    output_dir: Path,
    show: bool = False,
) -> list[Path]:
    """Extra plots derived from per-level and efficiency data.

    Args:
        level_breakdown: Output of scripts.analyze.compute_level_breakdown().
        cost_efficiency: Output of scripts.analyze.compute_cost_efficiency().
        output_dir:      Directory to save PNGs.
        show:            Display plots interactively.

    Returns list of written file paths.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    levels = sorted({lvl for wf in level_breakdown.values() for lvl in wf})
    workflows = sorted(level_breakdown.keys())
    n = len(workflows)
    colors = cm.tab10(np.linspace(0, 0.9, n))  # type: ignore[arg-type]

    # ── 5. Level breakdown grouped bar ────────────────────────────────────────
    if levels and workflows:
        n_levels = len(levels)
        x = np.arange(n_levels)
        # Ensure bars are at least 0.08 wide; scale figure width so they stay readable
        bar_width = max(0.08, 0.7 / max(n, 1))
        fig_width = max(10, n_levels * (n * bar_width + 1.0) + 2)
        label_fontsize = 7 if n > 6 else 8

        fig, ax = plt.subplots(figsize=(fig_width, 5))
        for i, (wf, c) in enumerate(zip(workflows, colors)):
            heights = [
                level_breakdown[wf].get(lvl, {}).get("success_rate", 0.0) * 100
                for lvl in levels
            ]
            offset = (i - n / 2 + 0.5) * bar_width
            ax.bar(x + offset, heights, bar_width * 0.9, label=wf, color=c, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{lvl}\n({LEVEL_NAMES.get(lvl, '')})" for lvl in levels], fontsize=9
        )
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Success Rate by Task Level", fontweight="bold")
        ax.legend(
            loc="upper right", bbox_to_anchor=(1.0 + 0.22 * max(1, n / 5), 1.02),
            fontsize=label_fontsize, framealpha=0.9,
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = output_dir / "level_breakdown.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        written.append(p)
        if show:
            plt.show()
        plt.close(fig)

    # ── 6. Cost-efficiency (quality / 1k tokens) ──────────────────────────────
    eff_data = [
        (wf, cost_efficiency[wf].get("quality_per_1k_tokens", 0.0) or 0.0)
        for wf in workflows
        if wf in cost_efficiency
    ]
    if eff_data:
        eff_data_sorted = sorted(eff_data, key=lambda x: x[1])
        eff_names = [d[0] for d in eff_data_sorted]
        eff_vals = [d[1] for d in eff_data_sorted]
        # Use wf_name as the index key, not the loop variable 'n' (which shadows len(workflows))
        eff_colors = [colors[workflows.index(wf_name)] for wf_name in eff_names]

        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(eff_names) + 1)))
        bars = ax.barh(eff_names, eff_vals, color=eff_colors, edgecolor="white")
        ax.set_xlabel("Quality Score per 1 000 Tokens (higher = more efficient)")
        ax.set_title("Token Efficiency: Quality per 1k Tokens", fontweight="bold")
        for bar, val in zip(bars, eff_vals):
            ax.text(
                val + max(eff_vals) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9,
            )
        if eff_vals:
            ax.axvline(
                x=np.mean(eff_vals), color="grey", linestyle="--", linewidth=1, label="mean"
            )
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = output_dir / "cost_efficiency.png"
        fig.savefig(p, dpi=150)
        written.append(p)
        if show:
            plt.show()
        plt.close(fig)

    return written


def main(
    file: str | None = None,
    out: str = "plots",
    show: bool = False,
) -> None:
    from config import settings

    results_dir = settings.results_dir
    plots_dir = Path(out)

    if file:
        summaries = _load_summary(Path(file))
    else:
        try:
            summaries = _load_latest_summary(results_dir)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if not summaries:
        print("No workflow summaries found in the results file.", file=sys.stderr)
        sys.exit(1)

    print(f"Plotting {len(summaries)} workflow(s)...")
    written = plot_all(summaries, plots_dir, show=show)
    for p in written:
        print(f"  Saved: {p}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--file", help="Path to a specific summary JSON file")
    parser.add_argument("--out", default="plots", help="Output directory for plots (default: plots/)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()
    main(file=args.file, out=args.out, show=args.show)
