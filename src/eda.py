"""Corpus Exploratory Data Analysis (EDA) functions."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_corpus_stats(raw_sentences):
    """Compute summary statistics for a raw corpus.

    Returns:
        stats (dict): Dictionary with 'sentences', 'tokens', 'vocabulary', 'mean_length'.
        raw_counts (Counter): Frequency of each raw token.
        lengths (np.ndarray): Array of sentence lengths.
    """
    raw_counts = Counter(w for s in raw_sentences for w in s)
    lengths = np.array([len(s) for s in raw_sentences])
    stats = {
        "sentences": len(raw_sentences),
        "tokens": int(lengths.sum()) if len(lengths) else 0,
        "vocabulary": len(raw_counts),
        "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
    }
    return stats, raw_counts, lengths


def plot_eda(lengths, raw_counts, output_path=None):
    """Plot sentence length histogram and Zipf's law log-log rank-frequency curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(lengths, bins=60, color="#245f7a")
    axes[0].set(
        xlabel="Panjang kalimat raw (token)",
        ylabel="Jumlah kalimat",
        title="Distribusi panjang",
    )

    ranks = np.arange(1, len(raw_counts) + 1)
    frequencies = sorted(raw_counts.values(), reverse=True)
    axes[1].loglog(ranks, frequencies, color="#245f7a")
    axes[1].set(
        xlabel="Rank",
        ylabel="Frekuensi",
        title="Rank-frequency raw (log-log)",
    )
    fig.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig, axes
