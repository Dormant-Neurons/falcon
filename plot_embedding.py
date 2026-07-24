import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from umap import UMAP

COLOR_MAP_DEFAULT = {
    "benign": "#8c8c8c",
}


def build_color_map(labels):
    uniq = sorted(set(labels) - set(COLOR_MAP_DEFAULT))
    n = len(uniq)
    cmap = dict(COLOR_MAP_DEFAULT)

    if n == 0:
        return cmap

    if n <= 10:
        base = plt.get_cmap("tab10")
        colors = [base(i) for i in range(n)]
    elif n <= 20:
        base = plt.get_cmap("tab20")
        colors = [base(i) for i in range(n)]
    else:
        base = plt.get_cmap("gist_ncar")
        colors = [base(i / max(n - 1, 1)) for i in range(n)]

    for cls, color in zip(uniq, colors):
        cmap[cls] = mcolors.to_hex(color)

    return cmap


def fit_projection(z, method="umap", random_state=42):
    if method == "umap":
        reducer = UMAP(n_components=2, random_state=random_state)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(z) // 20)),
            random_state=random_state,
            init="pca",
        )
    return reducer.fit_transform(z)


def plot_embedding(z, labels, title="embedding", method="umap",
                   max_per_family=100, out_path="last_round.png",
                   random_state=42):
    z = np.asarray(z)
    labels = np.asarray(labels)

    rng = np.random.default_rng(random_state)
    chosen_idx = []
    for cls in sorted(set(labels)):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) > max_per_family:
            cls_idx = rng.choice(cls_idx, size=max_per_family, replace=False)
        chosen_idx.append(cls_idx)
    chosen_idx = np.concatenate(chosen_idx)
    z, labels = z[chosen_idx], labels[chosen_idx]

    print(f"stratified subsample: {len(set(labels))} family, "
          f"{max_per_family} sample per family "
          f"(total {len(z)} samples)")

    print(f"{z.shape[0]} embedding, {method} to 2D")
    proj = fit_projection(z, method=method, random_state=random_state)

    color_map = build_color_map(labels)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cls in sorted(set(labels)):
        mask = labels == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   s=30, alpha=0.85,
                   c=color_map.get(cls, "#333333"),
                   marker="o" if cls == "benign" else "x",
                   linewidths=1.3,
                   label=cls)

    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    abs_path = os.path.abspath(out_path)
    plt.savefig(abs_path, dpi=200, bbox_inches="tight")

    logging.info("=" * 60)
    logging.info("Embedding plot saved.")
    logging.info("Current working directory: %s", os.getcwd())
    logging.info("Saved to: %s", abs_path)
    logging.info("File exists: %s", os.path.exists(abs_path))

    logging.info("=" * 60)
    return fig, ax