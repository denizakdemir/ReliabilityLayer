"""
Regenerate the two domain-level benchmark figures from the benchmark CSVs.

Previously these PNGs were produced by hand from the CSVs; this script makes
them reproducible.

Usage:
    python experiments/make_benchmark_figures.py \
        --results-dir experiments/results --out-dir experiments/figures
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ALPHA = 0.10

SPECS = [
    # (csv stem, figure stem, title, calibration domain label)
    ("wilds_camelyon_fallback_domains", "camelyon_fallback_domain_metrics",
     "UCI Covertype by Wilderness Area", "calibration domain: wilderness area 1"),
    ("wilds_amazon_fallback_domains", "amazon_fallback_domain_metrics",
     "20 Newsgroups by topic group", "calibration domain: comp"),
]


def make(csv_path, png_path, title, note):
    df = pd.read_csv(csv_path)
    names = df["domain_value"].astype(str).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [
        ("coverage", "Empirical coverage", 1.0 - ALPHA),
        ("delta", r"Deficiency estimate $\hat\delta$", None),
        ("ece", "ECE", None),
    ]
    for ax, (col, label, ref) in zip(axes, panels):
        ax.bar(names, df[col], color="#4C72B0")
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=1)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle(f"{title} ({note})", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--out-dir", default="experiments/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for csv_stem, fig_stem, title, note in SPECS:
        out = os.path.join(args.out_dir, fig_stem + ".png")
        make(os.path.join(args.results_dir, csv_stem + ".csv"), out, title, note)
        print("wrote", out)


if __name__ == "__main__":
    main()
