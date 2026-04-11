"""Linear probe training and evaluation for head-level analysis."""

from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.utils.logging import log


def extract_head_activations(
    hidden_final: np.ndarray,
    layer: int,
    head_idx: int,
    head_dim: int,
) -> np.ndarray:
    """Extract a single attention head's activation from the full hidden state.

    Args:
        hidden_final: (n_layers, hidden_dim)
        layer: layer index
        head_idx: head index within the layer
        head_dim: dimension per head

    Returns:
        (head_dim,) activation vector
    """
    start = head_idx * head_dim
    end = start + head_dim
    return hidden_final[layer, start:end]


def collect_head_features(
    hidden_finals: list[np.ndarray],
    layer: int,
    head_idx: int,
    head_dim: int,
) -> np.ndarray:
    """Collect head activations across all items for probing.

    Returns:
        X: (n_items, head_dim)
    """
    features = [
        extract_head_activations(hf, layer, head_idx, head_dim)
        for hf in hidden_finals
    ]
    return np.stack(features, axis=0)


def collect_layer_features(
    hidden_finals: list[np.ndarray],
    layer: int,
    pca_components: Optional[int] = 50,
) -> np.ndarray:
    """Collect full-layer activations across items, optionally with PCA.

    Args:
        hidden_finals: list of (n_layers, hidden_dim)
        layer: layer index
        pca_components: if set, reduce to this many components

    Returns:
        X: (n_items, n_features)
    """
    features = np.stack([hf[layer] for hf in hidden_finals], axis=0)
    if pca_components is not None and pca_components < features.shape[1]:
        pca = PCA(n_components=pca_components)
        features = pca.fit_transform(features)
    return features


def train_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    probe_type: str = "ridge",
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Train a linear probe with stratified cross-validation.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) integer labels
        n_folds: number of CV folds
        probe_type: 'ridge' or 'logistic'
        alpha: regularization strength

    Returns:
        dict with mean_accuracy, std_accuracy, fold_accuracies, n_samples, n_classes
    """
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    if n_classes < 2:
        return {
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "fold_accuracies": [],
            "n_samples": len(y),
            "n_classes": n_classes,
        }

    # Ensure enough samples per class for stratified k-fold
    min_per_class = min(np.sum(y == c) for c in unique_classes)
    actual_folds = min(n_folds, min_per_class)
    if actual_folds < 2:
        return {
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "fold_accuracies": [],
            "n_samples": len(y),
            "n_classes": n_classes,
        }

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    fold_accs: list[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if probe_type == "ridge":
            clf = RidgeClassifier(alpha=alpha)
        else:
            clf = LogisticRegression(
                C=1.0 / alpha, max_iter=1000, solver="lbfgs",
                multi_class="multinomial" if n_classes > 2 else "auto",
            )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        fold_accs.append(acc)

    return {
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": fold_accs,
        "n_samples": len(y),
        "n_classes": n_classes,
    }


def build_identity_labels(
    stimuli_items: list[dict],
    category_key: str = "category",
) -> tuple[np.ndarray, LabelEncoder]:
    """Build integer labels for identity category classification (Probe A).

    Returns:
        y: (n_items,) integer labels
        encoder: fitted LabelEncoder
    """
    le = LabelEncoder()
    raw_labels = [item[category_key] for item in stimuli_items]
    y = le.fit_transform(raw_labels)
    return y, le


def build_stereotyping_labels(
    stimuli_items: list[dict],
    metadatas: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build binary labels for stereotyping prediction (Probe B).

    Among ambiguous items where the model made a non-unknown prediction,
    label = 1 if model chose stereotyped target, 0 otherwise.

    Returns:
        mask: boolean array indicating which items to include
        y: binary labels for included items
    """
    mask = np.zeros(len(stimuli_items), dtype=bool)
    labels: list[int] = []

    for i, (item, meta) in enumerate(zip(stimuli_items, metadatas)):
        if item["context_condition"] != "ambig":
            continue

        # Determine model's prediction from logits
        logits = {}
        for letter in ["A", "B", "C"]:
            key = f"logit_{letter}"
            if key in meta:
                logits[letter] = meta[key]
        if not logits:
            continue

        pred_letter = max(logits, key=logits.get)
        pred_role = item["answer_roles"].get(pred_letter, "unknown")

        if pred_role == "unknown":
            continue  # model chose "can't determine" — skip

        mask[i] = True
        labels.append(1 if pred_role == "stereotyped_target" else 0)

    return mask, np.array(labels, dtype=np.int64)


def build_subgroup_labels(
    stimuli_items: list[dict],
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Build labels for within-category sub-group classification (Probe C).

    Returns:
        mask: boolean array indicating which items to include
        y: integer labels for included items
        encoder: fitted LabelEncoder
    """
    raw_labels: list[str] = []
    indices: list[int] = []

    for i, item in enumerate(stimuli_items):
        groups = item.get("stereotyped_groups", [])
        if groups:
            raw_labels.append(groups[0].lower())
            indices.append(i)

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)

    mask = np.zeros(len(stimuli_items), dtype=bool)
    mask[indices] = True

    return mask, y, le


def run_head_probes(
    hidden_finals: list[np.ndarray],
    labels: np.ndarray,
    mask: Optional[np.ndarray],
    n_layers: int,
    n_heads: int,
    head_dim: int,
    probe_type: str = "ridge",
    progress_every: int = 50,
) -> np.ndarray:
    """Run probe training across all layers and heads.

    Args:
        hidden_finals: list of (n_layers, hidden_dim)
        labels: integer labels (may be shorter than hidden_finals if mask is used)
        mask: optional boolean mask on items
        n_layers, n_heads, head_dim: model architecture
        probe_type: 'ridge' or 'logistic'
        progress_every: log progress every N heads

    Returns:
        accuracy_matrix: (n_layers, n_heads) of mean CV accuracies
    """
    if mask is not None:
        filtered_finals = [hf for hf, m in zip(hidden_finals, mask) if m]
    else:
        filtered_finals = hidden_finals

    accuracy_matrix = np.zeros((n_layers, n_heads), dtype=np.float32)
    total_heads = n_layers * n_heads
    done = 0

    for layer in range(n_layers):
        for head in range(n_heads):
            X = collect_head_features(filtered_finals, layer, head, head_dim)
            result = train_probe_cv(X, labels, probe_type=probe_type)
            accuracy_matrix[layer, head] = result["mean_accuracy"]

            done += 1
            if done % progress_every == 0:
                log(f"  [{done}/{total_heads}] heads probed")

    return accuracy_matrix
