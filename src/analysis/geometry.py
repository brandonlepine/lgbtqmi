"""Geometric analysis: cosine matrices, PCA, hierarchical clustering."""

from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

from src.utils.logging import log


def cosine_similarity_matrix(
    directions: dict[str, np.ndarray],
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise cosine similarity between directions at a given layer.

    Args:
        directions: dict mapping name -> (n_layers, hidden_dim)
        layer: layer index

    Returns:
        sim_matrix: (n, n) symmetric matrix with 1s on diagonal
        names: ordered list of direction names
    """
    names = sorted(directions.keys())
    n = len(names)
    vecs = np.stack([directions[name][layer] for name in names])  # (n, dim)

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vecs_normed = vecs / norms

    sim_matrix = vecs_normed @ vecs_normed.T
    return sim_matrix.astype(np.float32), names


def cosine_trajectory(
    dir_a: np.ndarray,
    dir_b: np.ndarray,
) -> np.ndarray:
    """Compute layer-wise cosine similarity between two directions.

    Args:
        dir_a, dir_b: (n_layers, hidden_dim)

    Returns:
        cosines: (n_layers,)
    """
    n_layers = dir_a.shape[0]
    cosines = np.zeros(n_layers, dtype=np.float32)
    for layer in range(n_layers):
        a = dir_a[layer]
        b = dir_b[layer]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            cosines[layer] = np.dot(a, b) / (na * nb)
    return cosines


def run_pca(
    directions: dict[str, np.ndarray],
    layer: int,
    n_components: int = 7,
) -> dict:
    """Run PCA on stacked category directions at a given layer.

    Returns dict with:
        'explained_variance_ratio': array of variance ratios
        'components': (n_components, hidden_dim)
        'loadings': (n_categories, n_components) — projections of directions
        'names': ordered list of category names
    """
    names = sorted(directions.keys())
    vecs = np.stack([directions[name][layer] for name in names])  # (n, dim)

    n_comp = min(n_components, len(names), vecs.shape[1])
    pca = PCA(n_components=n_comp)
    loadings = pca.fit_transform(vecs)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "loadings": loadings,
        "names": names,
        "n_components": n_comp,
    }


def hierarchical_clustering(
    directions: dict[str, np.ndarray],
    layer: int,
    method: str = "average",
) -> tuple[np.ndarray, list[str]]:
    """Hierarchical clustering of category directions using 1 - |cosine|.

    Returns:
        linkage_matrix: scipy linkage matrix
        names: ordered list of direction names
    """
    sim_matrix, names = cosine_similarity_matrix(directions, layer)
    # Distance = 1 - |cosine|
    dist_matrix = 1.0 - np.abs(sim_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    # Ensure symmetry and non-negativity
    dist_matrix = np.maximum(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=method)
    return Z, names


def cluster_ordering(
    directions: dict[str, np.ndarray],
    layer: int,
) -> list[str]:
    """Return names ordered by hierarchical clustering leaf order."""
    Z, names = hierarchical_clustering(directions, layer)
    order = leaves_list(Z)
    return [names[i] for i in order]


def shared_component_analysis(
    directions: dict[str, np.ndarray],
    layer: int,
) -> dict:
    """Extract shared and category-specific components via PCA.

    Returns dict with:
        'shared_direction': (hidden_dim,) — PC1, the shared marginalization axis
        'shared_projections': dict[name -> scalar] — projection magnitude
        'residuals': dict[name -> (hidden_dim,)] — after removing shared component
        'residual_cosine_matrix': (n, n) — cosines between residuals
        'variance_decomposition': dict[name -> {shared, meso, specific}]
    """
    pca_result = run_pca(directions, layer)
    names = pca_result["names"]
    loadings = pca_result["loadings"]
    components = pca_result["components"]
    var_ratios = pca_result["explained_variance_ratio"]

    shared_dir = components[0]  # PC1
    shared_dir = shared_dir / max(np.linalg.norm(shared_dir), 1e-8)

    # Project each direction onto shared axis
    shared_projections: dict[str, float] = {}
    residuals: dict[str, np.ndarray] = {}

    for i, name in enumerate(names):
        vec = directions[name][layer]
        proj = np.dot(vec, shared_dir)
        shared_projections[name] = float(proj)
        residuals[name] = vec - proj * shared_dir

    # Cosine matrix of residuals
    res_names = sorted(residuals.keys())
    n = len(res_names)
    res_vecs = np.stack([residuals[nm] for nm in res_names])
    res_norms = np.linalg.norm(res_vecs, axis=1, keepdims=True)
    res_norms = np.maximum(res_norms, 1e-8)
    res_normed = res_vecs / res_norms
    residual_cosines = res_normed @ res_normed.T

    # Variance decomposition per category
    variance_decomposition: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        vec = directions[name][layer]
        total_var = float(np.dot(vec, vec))
        shared_var = float(shared_projections[name] ** 2)
        # Meso = projection onto PC2+PC3
        meso_var = 0.0
        if components.shape[0] >= 3:
            for pc_idx in [1, 2]:
                pc = components[pc_idx]
                pc = pc / max(np.linalg.norm(pc), 1e-8)
                meso_var += float(np.dot(vec, pc) ** 2)
        specific_var = max(total_var - shared_var - meso_var, 0.0)
        total = max(shared_var + meso_var + specific_var, 1e-8)
        variance_decomposition[name] = {
            "shared": shared_var / total,
            "meso": meso_var / total,
            "specific": specific_var / total,
        }

    return {
        "shared_direction": shared_dir.astype(np.float32),
        "shared_projections": shared_projections,
        "residuals": {k: v.astype(np.float32) for k, v in residuals.items()},
        "residual_cosine_matrix": residual_cosines.astype(np.float32),
        "residual_names": res_names,
        "variance_decomposition": variance_decomposition,
        "pca_variance_ratios": var_ratios.tolist(),
    }
