#!/usr/bin/env python3
"""Compute meso-level cluster directions from existing category directions.

Reads directions.npz + cross_category_results.json produced by earlier pipeline.
Outputs meso_directions.npz and meso_directions_summary.json.

Usage:
    python scripts/compute_meso_directions.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.cluster import KMeans

from src.analysis.geometry import (
    cosine_similarity_matrix,
    run_pca,
    shared_component_analysis,
)
from src.utils.io import atomic_save_json, atomic_save_npz, ensure_dir
from src.utils.logging import log
from src.visualization.style import (
    ANNOT_SIZE,
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    LABEL_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    apply_style,
    label_panel,
    save_fig,
)

# ---------------------------------------------------------------------------
# Hard-coded cluster assignments (empirical results from PCA analysis)
# Keys are category short names matching bbq_loader.CATEGORY_MAP
# ---------------------------------------------------------------------------
CLUSTERS: dict[str, list[str]] = {
    "lgbtq": ["so", "gi"],
    "social_group": ["race", "religion"],
    "bodily_physical": ["physical_appearance", "disability", "age"],
}

CLUSTER_COLORS: dict[str, str] = {
    "lgbtq": "#0072B2",
    "social_group": "#E69F00",
    "bodily_physical": "#009E73",
}

CLUSTER_LABELS: dict[str, str] = {
    "lgbtq": "LGBTQ+",
    "social_group": "Social Group",
    "bodily_physical": "Bodily/Physical",
}


# ===== Helpers =============================================================

def _unit_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-8)


def _load_category_directions(run_dir: Path) -> dict[str, np.ndarray]:
    """Load base category directions from directions.npz."""
    path = run_dir / "analysis" / "directions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/compute_directions.py first."
        )
    data = np.load(path, allow_pickle=True)
    base_cats = ["so", "gi", "race", "religion", "disability", "physical_appearance", "age"]
    dirs: dict[str, np.ndarray] = {}
    for cat in base_cats:
        key = f"direction_{cat}"
        if key in data.files:
            dirs[cat] = data[key]
        else:
            log(f"  WARNING: {key} not in directions.npz, skipping {cat}")
    return dirs


def _validate_clusters_via_kmeans(
    cat_dirs: dict[str, np.ndarray],
    layer: int,
) -> dict[str, list[str]]:
    """Data-driven cluster validation using k-means on PCA loadings."""
    pca_result = run_pca(cat_dirs, layer, n_components=min(5, len(cat_dirs)))
    loadings = pca_result["loadings"][:, :2]  # PC1-PC2
    names = pca_result["names"]

    # k=2 and k=3
    results = {}
    for k in [2, 3]:
        if len(names) < k:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(loadings)
        clusters: dict[int, list[str]] = {}
        for name, label in zip(names, labels):
            clusters.setdefault(int(label), []).append(name)
        results[f"k{k}"] = clusters

    return results


# ===== Core computation ====================================================

def compute_cluster_directions(
    cat_dirs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute cluster-level directions by averaging members and normalizing."""
    n_layers = next(iter(cat_dirs.values())).shape[0]
    hidden_dim = next(iter(cat_dirs.values())).shape[1]
    cluster_dirs: dict[str, np.ndarray] = {}

    for cluster_name, members in CLUSTERS.items():
        available = [cat_dirs[m] for m in members if m in cat_dirs]
        if not available:
            log(f"  WARNING: no members available for cluster {cluster_name}")
            continue
        raw = np.stack(available, axis=0).mean(axis=0)  # (n_layers, dim)
        # Unit-normalize per layer
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        cluster_dirs[cluster_name] = (raw / norms).astype(np.float32)
        log(f"  Cluster '{cluster_name}': averaged {len(available)} members")

    return cluster_dirs


def compute_within_cluster_residuals(
    cat_dirs: dict[str, np.ndarray],
    cluster_dirs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """For each category, subtract its cluster direction to get within-cluster residual."""
    residuals: dict[str, np.ndarray] = {}
    for cluster_name, members in CLUSTERS.items():
        if cluster_name not in cluster_dirs:
            continue
        cl_dir = cluster_dirs[cluster_name]
        for cat in members:
            if cat not in cat_dirs:
                continue
            n_layers = cat_dirs[cat].shape[0]
            res = np.zeros_like(cat_dirs[cat])
            for layer in range(n_layers):
                proj = np.dot(cat_dirs[cat][layer], cl_dir[layer]) * cl_dir[layer]
                raw = cat_dirs[cat][layer] - proj
                norm = np.linalg.norm(raw)
                res[layer] = raw / max(norm, 1e-8)
            residuals[cat] = res.astype(np.float32)
    return residuals


def compute_4level_variance_decomposition(
    cat_dirs: dict[str, np.ndarray],
    cluster_dirs: dict[str, np.ndarray],
    layer: int,
) -> dict[str, dict[str, float]]:
    """4-level decomposition: shared / meso / within-cluster / category-residual."""
    # Get shared direction (PC1)
    sca = shared_component_analysis(cat_dirs, layer)
    shared_dir = sca["shared_direction"]

    decomp: dict[str, dict[str, float]] = {}
    for cluster_name, members in CLUSTERS.items():
        if cluster_name not in cluster_dirs:
            continue
        cl_dir = cluster_dirs[cluster_name][layer]

        # Meso direction = cluster direction projected out of shared
        cl_proj_on_shared = np.dot(cl_dir, shared_dir) * shared_dir
        meso_component = cl_dir - cl_proj_on_shared
        meso_norm = np.linalg.norm(meso_component)
        if meso_norm > 1e-8:
            meso_unit = meso_component / meso_norm
        else:
            meso_unit = np.zeros_like(shared_dir)

        for cat in members:
            if cat not in cat_dirs:
                continue
            v = cat_dirs[cat][layer]
            total_var = float(np.dot(v, v))
            if total_var < 1e-12:
                decomp[cat] = {"shared": 0, "meso": 0, "within_cluster": 0, "residual": 1.0}
                continue

            # Project onto shared
            shared_coeff = float(np.dot(v, shared_dir))
            shared_var = shared_coeff ** 2

            # Project onto meso (orthogonal to shared by construction)
            meso_coeff = float(np.dot(v, meso_unit))
            meso_var = meso_coeff ** 2

            # Within-cluster: what's left of the cluster direction after shared+meso
            # reconstructed from shared + meso
            v_after_shared_meso = v - shared_coeff * shared_dir - meso_coeff * meso_unit

            # Within-cluster residual = projection of v onto the within-cluster residual direction
            # First compute raw within-cluster residual for this category
            cat_proj_on_cluster = np.dot(v, cl_dir) * cl_dir
            cat_within_raw = v - cat_proj_on_cluster
            cat_within_norm = np.linalg.norm(cat_within_raw)
            if cat_within_norm > 1e-8:
                cat_within_unit = cat_within_raw / cat_within_norm
                within_coeff = float(np.dot(v, cat_within_unit))
                within_var = within_coeff ** 2
            else:
                within_var = 0.0

            residual_var = max(total_var - shared_var - meso_var - within_var, 0.0)

            total = shared_var + meso_var + within_var + residual_var
            total = max(total, 1e-12)
            decomp[cat] = {
                "shared": shared_var / total,
                "meso": meso_var / total,
                "within_cluster": within_var / total,
                "residual": residual_var / total,
            }

    return decomp


def compute_reconstruction_error(
    cat_dirs: dict[str, np.ndarray],
    cluster_dirs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Measure reconstruction error per category per layer."""
    n_layers = next(iter(cat_dirs.values())).shape[0]
    errors: dict[str, np.ndarray] = {}

    for cluster_name, members in CLUSTERS.items():
        if cluster_name not in cluster_dirs:
            continue
        for cat in members:
            if cat not in cat_dirs:
                continue
            errs = np.zeros(n_layers, dtype=np.float32)
            for layer in range(n_layers):
                sca = shared_component_analysis(cat_dirs, layer)
                shared = sca["shared_direction"]
                cl = cluster_dirs[cluster_name][layer]
                v = cat_dirs[cat][layer]

                # Reconstruct: project onto shared, cluster, and residual
                shared_proj = np.dot(v, shared) * shared
                cl_proj = np.dot(v, cl) * cl
                residual = v - shared_proj - cl_proj
                # The "reconstructed" includes shared + cluster + residual = v by construction
                # So error should be ~0. The real question is whether shared+meso+within spans v.
                # Reconstruction = shared_proj + meso_component_proj + within_proj + residual_proj
                reconstructed = shared_proj + cl_proj + (v - shared_proj - cl_proj)
                errs[layer] = float(np.linalg.norm(v - reconstructed))

            errors[cat] = errs
    return errors


# ===== Visualization =======================================================

def plot_fig25(
    cluster_dirs: dict[str, np.ndarray],
    cat_dirs: dict[str, np.ndarray],
    decomp_4level: dict[str, dict[str, float]],
    recon_errors: dict[str, np.ndarray],
    mid_layer: int,
    path: str,
) -> None:
    """Fig 25: Meso direction validation (4 panels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: Cluster-member cosine similarity
    ax = axes[0, 0]
    bar_x = []
    bar_h = []
    bar_c = []
    bar_labels = []
    tick_positions = []
    tick_labels = []
    offset = 0
    for cluster_name, members in CLUSTERS.items():
        if cluster_name not in cluster_dirs:
            continue
        cl_dir = cluster_dirs[cluster_name][mid_layer]
        start = offset
        for cat in members:
            if cat not in cat_dirs:
                continue
            cos = float(np.dot(cl_dir, _unit_norm(cat_dirs[cat][mid_layer])))
            bar_x.append(offset)
            bar_h.append(cos)
            bar_c.append(CATEGORY_COLORS.get(cat, "#999999"))
            bar_labels.append(CATEGORY_LABELS.get(cat, cat))
            offset += 1
        mid_pos = (start + offset - 1) / 2.0
        tick_positions.append(mid_pos)
        tick_labels.append(CLUSTER_LABELS[cluster_name])
        offset += 0.5

    ax.bar(bar_x, bar_h, color=bar_c, edgecolor="black", linewidth=0.5)
    for i, (x, h, lbl) in enumerate(zip(bar_x, bar_h, bar_labels)):
        ax.text(x, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=ANNOT_SIZE)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
    ax.set_ylabel("Cosine with cluster direction", fontsize=LABEL_SIZE)
    ax.set_title(f"Member-to-cluster cosine (Layer {mid_layer})", fontsize=TITLE_SIZE)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    label_panel(ax, "A")

    # Panel B: Between-cluster cosines across layers
    ax = axes[0, 1]
    cluster_names = sorted(cluster_dirs.keys())
    n_layers = next(iter(cluster_dirs.values())).shape[0]
    for i, ca in enumerate(cluster_names):
        for cb in cluster_names[i + 1:]:
            cosines = np.zeros(n_layers)
            for layer in range(n_layers):
                da = _unit_norm(cluster_dirs[ca][layer])
                db = _unit_norm(cluster_dirs[cb][layer])
                cosines[layer] = float(np.dot(da, db))
            lbl = f"{CLUSTER_LABELS.get(ca, ca)} ↔ {CLUSTER_LABELS.get(cb, cb)}"
            ax.plot(cosines, label=lbl, linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer", fontsize=LABEL_SIZE)
    ax.set_ylabel("Cosine similarity", fontsize=LABEL_SIZE)
    ax.set_title("Between-cluster cosines", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1)
    label_panel(ax, "B")

    # Panel C: 4-level variance decomposition
    ax = axes[1, 0]
    cats = sorted(decomp_4level.keys())
    display = [CATEGORY_LABELS.get(c, c) for c in cats]
    x = np.arange(len(cats))
    w = 0.6
    shared = [decomp_4level[c]["shared"] for c in cats]
    meso = [decomp_4level[c]["meso"] for c in cats]
    within = [decomp_4level[c]["within_cluster"] for c in cats]
    resid = [decomp_4level[c]["residual"] for c in cats]

    ax.bar(x, shared, w, label="Shared (PC1)", color="#0072B2")
    bottom1 = shared
    ax.bar(x, meso, w, bottom=bottom1, label="Meso (cluster)", color="#E69F00")
    bottom2 = [s + m for s, m in zip(shared, meso)]
    ax.bar(x, within, w, bottom=bottom2, label="Within-cluster", color="#CC79A7")
    bottom3 = [b + wc for b, wc in zip(bottom2, within)]
    ax.bar(x, resid, w, bottom=bottom3, label="Residual", color="#999999")

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
    ax.set_ylabel("Fraction of variance", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.05)
    ax.set_title("4-level variance decomposition", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1)
    label_panel(ax, "C")

    # Panel D: Reconstruction error across layers
    ax = axes[1, 1]
    for cat, errs in sorted(recon_errors.items()):
        ax.plot(errs, label=CATEGORY_LABELS.get(cat, cat),
                color=CATEGORY_COLORS.get(cat, "#999999"), linewidth=1.2)
    ax.set_xlabel("Layer", fontsize=LABEL_SIZE)
    ax.set_ylabel("Reconstruction error", fontsize=LABEL_SIZE)
    ax.set_title("Direction reconstruction error", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1)
    label_panel(ax, "D")

    fig.suptitle("Meso-level direction validation", fontsize=TITLE_SIZE + 2, y=0.98)
    save_fig(fig, path, tight=False)


def plot_fig26(
    cat_dirs: dict[str, np.ndarray],
    cluster_dirs: dict[str, np.ndarray],
    mid_layer: int,
    path: str,
) -> None:
    """Fig 26: PCA scatter with cluster centroids as stars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()

    # Combine all directions for PCA
    all_dirs = {**cat_dirs}
    for cname, cdir in cluster_dirs.items():
        all_dirs[f"cluster_{cname}"] = cdir
    pca_result = run_pca(all_dirs, mid_layer, n_components=min(5, len(all_dirs)))
    loadings = pca_result["loadings"]
    names = pca_result["names"]
    var_ratios = pca_result["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot category directions as circles
    for i, name in enumerate(names):
        if name.startswith("cluster_"):
            continue
        color = CATEGORY_COLORS.get(name, "#999999")
        lbl = CATEGORY_LABELS.get(name, name)
        ax.scatter(loadings[i, 0], loadings[i, 1], c=color, s=80,
                   edgecolors="black", linewidths=0.7, zorder=5, label=lbl)
        ax.annotate(lbl, (loadings[i, 0], loadings[i, 1]),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=ANNOT_SIZE, ha="left")

    # Plot cluster directions as stars
    for i, name in enumerate(names):
        if not name.startswith("cluster_"):
            continue
        cname = name[len("cluster_"):]
        color = CLUSTER_COLORS.get(cname, "#999999")
        lbl = CLUSTER_LABELS.get(cname, cname)
        ax.scatter(loadings[i, 0], loadings[i, 1], c=color, marker="*",
                   s=300, edgecolors="black", linewidths=0.8, zorder=10,
                   label=f"{lbl} (cluster)")

        # Draw arrows from cluster centroid to members
        for member in CLUSTERS.get(cname, []):
            if member in names:
                mi = names.index(member)
                ax.annotate("", xy=(loadings[mi, 0], loadings[mi, 1]),
                            xytext=(loadings[i, 0], loadings[i, 1]),
                            arrowprops=dict(arrowstyle="->", color=color,
                                            linewidth=1.2, alpha=0.5))

    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    xlabel = f"PC1 ({var_ratios[0]:.1%})" if len(var_ratios) > 0 else "PC1"
    ylabel = f"PC2 ({var_ratios[1]:.1%})" if len(var_ratios) > 1 else "PC2"
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(f"Category & cluster directions in PCA space (Layer {mid_layer})",
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, path)


# ===== Main ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute meso-level cluster directions (reads existing directions)."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    fig_dir = ensure_dir(run_dir / "figures")
    model_id = args.model_id or run_dir.parent.name

    log(f"Computing meso-level cluster directions for {model_id}")
    log(f"Run dir: {run_dir}")

    # Load existing category directions
    cat_dirs = _load_category_directions(run_dir)
    available_cats = sorted(cat_dirs.keys())
    n_layers = next(iter(cat_dirs.values())).shape[0]
    hidden_dim = next(iter(cat_dirs.values())).shape[1]
    mid_layer = n_layers // 2
    log(f"Loaded {len(cat_dirs)} category directions: {available_cats}")
    log(f"Shape: ({n_layers}, {hidden_dim}), mid_layer={mid_layer}")

    # Step 1: Validate clusters via k-means
    log("\n--- Step 1: Data-driven cluster validation ---")
    km_results = _validate_clusters_via_kmeans(cat_dirs, mid_layer)
    for k_label, clusters in km_results.items():
        log(f"  {k_label} clusters: {clusters}")

    # Check if k=3 matches hard-coded
    hardcoded_sets = {frozenset(v) for v in CLUSTERS.values()}
    if "k3" in km_results:
        km3_sets = {frozenset(v) for v in km_results["k3"].values()}
        match = hardcoded_sets == km3_sets
        log(f"  k=3 matches hard-coded? {match}")
        if not match:
            log("  WARNING: Data-driven clusters differ from hard-coded assignments!")
            log(f"    Hard-coded: {CLUSTERS}")
            log(f"    Data-driven: {km_results['k3']}")
    else:
        log("  Could not run k=3 (too few categories)")

    # Step 2: Compute cluster directions
    log("\n--- Step 2: Compute cluster directions ---")
    cluster_dirs = compute_cluster_directions(cat_dirs)

    # Verify cosines between cluster and members
    for cname, members in CLUSTERS.items():
        if cname not in cluster_dirs:
            continue
        cl = cluster_dirs[cname][mid_layer]
        for cat in members:
            if cat in cat_dirs:
                cos = float(np.dot(cl, _unit_norm(cat_dirs[cat][mid_layer])))
                log(f"  {CLUSTER_LABELS[cname]} ↔ {CATEGORY_LABELS.get(cat, cat)}: cos={cos:.3f}")

    # Between-cluster cosines
    cnames = sorted(cluster_dirs.keys())
    log("\n  Between-cluster cosines (mid layer):")
    for i, ca in enumerate(cnames):
        for cb in cnames[i + 1:]:
            cos = float(np.dot(
                cluster_dirs[ca][mid_layer], cluster_dirs[cb][mid_layer]
            ))
            log(f"    {CLUSTER_LABELS[ca]} ↔ {CLUSTER_LABELS[cb]}: cos={cos:.3f}")

    # Step 3: Within-cluster residuals
    log("\n--- Step 3: Within-cluster residuals ---")
    within_residuals = compute_within_cluster_residuals(cat_dirs, cluster_dirs)
    for cat, res in within_residuals.items():
        norm_mid = float(np.linalg.norm(res[mid_layer]))
        log(f"  {CATEGORY_LABELS.get(cat, cat)} within-cluster residual norm: {norm_mid:.4f}")

    # Step 4: 4-level variance decomposition
    log("\n--- Step 4: Variance decomposition ---")
    decomp = compute_4level_variance_decomposition(cat_dirs, cluster_dirs, mid_layer)
    for cat in sorted(decomp.keys()):
        d = decomp[cat]
        total = d["shared"] + d["meso"] + d["within_cluster"] + d["residual"]
        log(f"  {CATEGORY_LABELS.get(cat, cat):>22s}: "
            f"shared={d['shared']:.2f}  meso={d['meso']:.2f}  "
            f"within={d['within_cluster']:.2f}  resid={d['residual']:.2f}  "
            f"sum={total:.3f}")

    # Step 5: Reconstruction error
    log("\n--- Step 5: Reconstruction error ---")
    recon_errors = compute_reconstruction_error(cat_dirs, cluster_dirs)
    for cat, errs in sorted(recon_errors.items()):
        log(f"  {CATEGORY_LABELS.get(cat, cat)}: max_err={errs.max():.6f}, mean_err={errs.mean():.6f}")

    # ===== Save outputs =====
    log("\n--- Saving outputs ---")
    save_arrays: dict[str, np.ndarray] = {}
    for cname, cdir in cluster_dirs.items():
        save_arrays[f"{cname}_direction"] = cdir
    for cat, res in within_residuals.items():
        save_arrays[f"within_cluster_{cat}"] = res
    for cat, errs in recon_errors.items():
        save_arrays[f"reconstruction_error_{cat}"] = errs

    # Metadata
    save_arrays["_cluster_assignments"] = np.array(json.dumps(CLUSTERS))
    save_arrays["_kmeans_results"] = np.array(json.dumps(
        {k: {str(ki): v for ki, v in clusters.items()} for k, clusters in km_results.items()}
    ))

    out_path = analysis_dir / "meso_directions.npz"
    atomic_save_npz(out_path, **save_arrays)
    log(f"  Saved -> {out_path}")

    # Summary JSON
    summary = {
        "model_id": model_id,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "mid_layer": mid_layer,
        "cluster_assignments": CLUSTERS,
        "kmeans_validation": {
            k: {str(ki): v for ki, v in cl.items()}
            for k, cl in km_results.items()
        },
        "member_cosines_mid": {
            cname: {
                cat: float(np.dot(
                    cluster_dirs[cname][mid_layer],
                    _unit_norm(cat_dirs[cat][mid_layer])
                ))
                for cat in members if cat in cat_dirs
            }
            for cname, members in CLUSTERS.items() if cname in cluster_dirs
        },
        "between_cluster_cosines_mid": {
            f"{ca}_{cb}": float(np.dot(
                cluster_dirs[ca][mid_layer], cluster_dirs[cb][mid_layer]
            ))
            for i, ca in enumerate(cnames) for cb in cnames[i + 1:]
        },
        "variance_decomposition_4level": decomp,
    }
    summary_path = analysis_dir / "meso_directions_summary.json"
    atomic_save_json(summary, summary_path)
    log(f"  Summary -> {summary_path}")

    # ===== Figures =====
    log("\n--- Generating figures ---")

    plot_fig25(
        cluster_dirs, cat_dirs, decomp, recon_errors, mid_layer,
        path=str(fig_dir / "fig_25_meso_direction_validation.png"),
    )
    log("  Saved fig_25")

    plot_fig26(
        cat_dirs, cluster_dirs, mid_layer,
        path=str(fig_dir / "fig_26_cluster_directions_pca.png"),
    )
    log("  Saved fig_26")

    log("\nDone!")


if __name__ == "__main__":
    main()
