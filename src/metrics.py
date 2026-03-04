"""
metrics.py — Métriques pour la problématique P02

Métriques rapportées :
  - F1-score macro (métrique principale)
  - Accuracy
  - Gap de généralisation (F1_train - F1_val)
  - Sharpness du minimum (loss landscape)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Métriques de classification
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    label_names: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Calcule accuracy et F1-score macro.

    Parameters
    ----------
    labels : Étiquettes de référence
    predictions : Prédictions du modèle
    label_names : Noms des classes (pour l'affichage)
    verbose : Affiche le rapport de classification complet

    Returns
    -------
    dict contenant 'accuracy', 'f1_macro', 'f1_per_class'
    """
    labels = np.array(labels)
    predictions = np.array(predictions)

    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0).tolist()

    if verbose:
        print("\n── Rapport de classification ──")
        print(classification_report(
            labels, predictions,
            target_names=label_names or [str(i) for i in range(len(f1_per_class))],
            zero_division=0,
        ))
        print("Matrice de confusion :")
        print(confusion_matrix(labels, predictions))

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per_class,
    }


# ---------------------------------------------------------------------------
# Gap de généralisation (P02)
# ---------------------------------------------------------------------------

def generalization_gap(train_f1: float, val_f1: float) -> dict[str, float]:
    """
    Calcule le gap de généralisation entre train et validation.

    Un gap élevé indique du sur-apprentissage.
    L'objectif de P02 est d'étudier comment weight_decay et dropout
    réduisent ce gap.

    Returns
    -------
    dict avec 'gap' (absolu) et 'gap_pct' (relatif en %)
    """
    gap = train_f1 - val_f1
    gap_pct = (gap / train_f1 * 100) if train_f1 > 0 else 0.0
    return {
        "train_f1": train_f1,
        "val_f1": val_f1,
        "gap": gap,
        "gap_pct": gap_pct,
    }


# ---------------------------------------------------------------------------
# Sharpness du loss landscape (P02 — formule du cours)
# ---------------------------------------------------------------------------

def compute_sharpness(
    base_loss: float,
    perturbed_losses: list[float],
) -> float:
    """
    Calcule la sharpness selon la formule du cours :

        Sharpness = (1/N) Σ |L(θ + ε·d_i) - L(θ)|

    Un minimum pointu (sharpness élevée) est associé à une mauvaise
    généralisation. Un minimum plat est associé à une meilleure
    généralisation (Hochreiter & Schmidhuber, 1997).

    Parameters
    ----------
    base_loss : L(θ) — loss au point de convergence
    perturbed_losses : [L(θ + ε·d_i)] pour i = 1..N

    Returns
    -------
    sharpness : float
    """
    deviations = [abs(l - base_loss) for l in perturbed_losses]
    return float(np.mean(deviations))


# ---------------------------------------------------------------------------
# Résumé comparatif pour Optuna
# ---------------------------------------------------------------------------

def format_trial_summary(
    trial_number: int,
    params: dict,
    train_metrics: dict,
    val_metrics: dict,
) -> str:
    """Formate un résumé lisible d'un trial Optuna."""
    gap = generalization_gap(train_metrics["f1_macro"], val_metrics["f1_macro"])
    lines = [
        f"\n{'─'*55}",
        f"  Trial #{trial_number:03d}",
        f"  weight_decay = {params.get('weight_decay', '?'):.1e}  |  dropout = {params.get('dropout', '?'):.2f}",
        f"  F1 train = {train_metrics['f1_macro']:.4f}  |  F1 val = {val_metrics['f1_macro']:.4f}",
        f"  Gap = {gap['gap']:.4f} ({gap['gap_pct']:.1f}%)",
        f"{'─'*55}",
    ]
    return "\n".join(lines)
