"""
visualization.py — Visualisations pour la problématique P02

Figures produites :
  1. Loss landscape 1D (minimum plat vs pointu)
  2. Heatmap gap de généralisation (weight_decay × dropout)
  3. Courbes de convergence (train vs val F1 par époque)
  4. Importance des hyperparamètres (depuis étude Optuna)
  5. Scatter plot : sharpness vs F1-val (relation généralisation)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.config import FIGURES_DIR, LandscapeConfig, RESULTS_DIR
from src.metrics import compute_sharpness

logger = logging.getLogger(__name__)

# Style global
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PALETTE = sns.color_palette("husl", 8)


# ---------------------------------------------------------------------------
# 1. Loss Landscape 1D
# ---------------------------------------------------------------------------

def compute_loss_landscape_1d(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    cfg: LandscapeConfig = LandscapeConfig(),
) -> tuple[np.ndarray, list[float]]:
    """
    Perturbation 1D du loss landscape autour du point de convergence θ*.

    Algorithme :
      1. Sauvegarder θ* (paramètres actuels)
      2. Tirer une direction aléatoire d normalisée
      3. Évaluer L(θ* + α·d) pour α ∈ [-ε, +ε]
      4. Restaurer θ*

    Returns
    -------
    alphas : np.ndarray (abscisses)
    losses : list[float] (ordonnées)
    """
    model.eval()
    original_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire normalisée (filter-normalized comme Li et al. 2018)
    direction = []
    for p in original_params:
        d = torch.randn_like(p)
        norm_p = p.norm()
        norm_d = d.norm()
        if norm_d > 0 and norm_p > 0:
            d = d * (norm_p / norm_d)   # Filter normalization
        direction.append(d)

    alphas = np.linspace(-cfg.epsilon, cfg.epsilon, cfg.n_points)
    losses = []

    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    n_batches = max(1, cfg.n_samples_eval // 16)

    for alpha in alphas:
        # Appliquer la perturbation
        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + float(alpha) * d.to(p0.device)

        # Évaluer sur un sous-ensemble
        batch_losses = []
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            batch_losses.append(outputs.loss.item())

        losses.append(float(np.mean(batch_losses)))

    # Restaurer θ*
    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0.to(p.device)

    return alphas, losses


def plot_loss_landscape_comparison(
    results: dict[str, tuple[np.ndarray, list[float]]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Compare les loss landscapes de plusieurs configurations.
    Typiquement : {dropout=0.0, dropout=0.1, dropout=0.3} ou
                  {wd=1e-5, wd=1e-3} etc.

    Parameters
    ----------
    results : dict label → (alphas, losses)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    styles = [
        {"color": PALETTE[0], "lw": 2.5, "ls": "-",  "marker": "o"},
        {"color": PALETTE[2], "lw": 2.5, "ls": "--", "marker": "s"},
        {"color": PALETTE[4], "lw": 2.5, "ls": ":",  "marker": "^"},
        {"color": PALETTE[6], "lw": 2.0, "ls": "-.", "marker": "D"},
    ]

    sharpnesses = {}
    for i, (label, (alphas, losses)) in enumerate(results.items()):
        s = styles[i % len(styles)]
        base_loss = losses[len(losses) // 2]   # Loss au centre (α≈0)
        sharpness = compute_sharpness(base_loss, losses)
        sharpnesses[label] = sharpness

        ax.plot(
            alphas, losses,
            label=f"{label}  (sharpness={sharpness:.4f})",
            color=s["color"], linewidth=s["lw"],
            linestyle=s["ls"], marker=s["marker"],
            markersize=5, markerfacecolor="white",
        )
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.set_xlabel("Direction de perturbation (α)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        "Loss Landscape 1D — CamemBERT sur Allociné\n"
        "Impact du weight decay et du dropout (G10 — P02)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.annotate(
        "← Minimum plat = meilleure généralisation →",
        xy=(0, ax.get_ylim()[0]),
        xytext=(0, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05),
        ha="center", fontsize=9, color="gray", style="italic",
    )

    plt.tight_layout()
    _save_or_show(fig, save_path, "loss_landscape_1d.png")
    return sharpnesses


# ---------------------------------------------------------------------------
# 2. Heatmap gap de généralisation
# ---------------------------------------------------------------------------

def plot_generalization_heatmap(
    df: pd.DataFrame,
    metric: str = "gap_train_val",
    save_path: Optional[Path] = None,
) -> None:
    """
    Heatmap des métriques P02 (gap train/val, gap train/test, F1-val) en fonction de
    weight_decay (lignes) et dropout (colonnes).
    """
    pivot = df.pivot(index="weight_decay", columns="dropout", values=metric)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Heatmap gap train/val (rouge = sur-apprentissage)
    sns.heatmap(
        df.pivot(index="weight_decay", columns="dropout", values="gap_train_val"),
        ax=axes[0],
        cmap="Reds",
        annot=True, fmt=".3f",
        linewidths=0.5,
        xticklabels=[f"{d:.1f}" for d in sorted(df["dropout"].unique())],
        yticklabels=[f"{w:.0e}" for w in sorted(df["weight_decay"].unique())],
    )
    axes[0].set_title("Écart train/val\n(F1_train − F1_val) — plus faible = mieux", fontsize=11)
    axes[0].set_xlabel("Dropout", fontsize=11)
    axes[0].set_ylabel("Weight Decay", fontsize=11)

    # Heatmap gap train/test (protocole P02 : "mesurer l'écart train/test")
    sns.heatmap(
        df.pivot(index="weight_decay", columns="dropout", values="gap_train_test"),
        ax=axes[1],
        cmap="Oranges",
        annot=True, fmt=".3f",
        linewidths=0.5,
        xticklabels=[f"{d:.1f}" for d in sorted(df["dropout"].unique())],
        yticklabels=[f"{w:.0e}" for w in sorted(df["weight_decay"].unique())],
    )
    axes[1].set_title("Écart train/test\n(F1_train − F1_test) — plus faible = mieux", fontsize=11)
    axes[1].set_xlabel("Dropout", fontsize=11)
    axes[1].set_ylabel("Weight Decay", fontsize=11)

    # Heatmap F1-val (vert = bon)
    sns.heatmap(
        df.pivot(index="weight_decay", columns="dropout", values="val_f1"),
        ax=axes[2],
        cmap="Greens",
        annot=True, fmt=".3f",
        linewidths=0.5,
        xticklabels=[f"{d:.1f}" for d in sorted(df["dropout"].unique())],
        yticklabels=[f"{w:.0e}" for w in sorted(df["weight_decay"].unique())],
    )
    axes[2].set_title("F1-score macro (validation)\n— plus élevé = mieux", fontsize=11)
    axes[2].set_xlabel("Dropout", fontsize=11)
    axes[2].set_ylabel("Weight Decay", fontsize=11)

    fig.suptitle(
        "Protocole P02 — Régularisation × Généralisation\n"
        "CamemBERT-base sur Allociné (G10)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, save_path, "heatmap_p02.png")


# ---------------------------------------------------------------------------
# 3. Courbes de convergence
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    histories: dict[str, dict],
    save_path: Optional[Path] = None,
) -> None:
    """
    Trace les courbes train/val F1 et loss pour plusieurs configurations.

    Parameters
    ----------
    histories : dict label → history dict (tel que retourné par CamembertTrainer)
    """
    n_configs = len(histories)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, (label, history) in enumerate(histories.items()):
        color = PALETTE[i % len(PALETTE)]
        epochs = range(1, len(history["train_f1"]) + 1)

        # F1-score
        axes[0].plot(epochs, history["train_f1"], color=color,
                     lw=2, ls="-", label=f"{label} (train)", alpha=0.9)
        axes[0].plot(epochs, history["val_f1"], color=color,
                     lw=2, ls="--", label=f"{label} (val)", alpha=0.9)

        # Loss
        axes[1].plot(epochs, history["train_loss"], color=color,
                     lw=2, ls="-", alpha=0.9)
        axes[1].plot(epochs, history["val_loss"], color=color,
                     lw=2, ls="--", alpha=0.9)

    axes[0].set_title("F1-score macro par époque", fontsize=12)
    axes[0].set_xlabel("Époque", fontsize=11)
    axes[0].set_ylabel("F1-score macro", fontsize=11)
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    axes[1].set_title("Loss par époque", fontsize=12)
    axes[1].set_xlabel("Époque", fontsize=11)
    axes[1].set_ylabel("Cross-entropy loss", fontsize=11)
    # Légende : trait plein = train, pointillé = val
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", lw=2, ls="-",  label="Train"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="Validation"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=10)

    fig.suptitle(
        "Courbes de convergence — CamemBERT / Allociné (G10 — P02)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, save_path, "convergence_curves.png")


# ---------------------------------------------------------------------------
# 4. Importance des hyperparamètres (Optuna)
# ---------------------------------------------------------------------------

def plot_optuna_importance(
    study,
    save_path: Optional[Path] = None,
) -> None:
    """Visualise l'importance des hyperparamètres selon Optuna."""
    try:
        from optuna.visualization.matplotlib import plot_param_importances

        # Avec Optuna >= 3.x, plot_param_importances ne prend plus "ax" en argument
        # et renvoie un objet Axes. On récupère ensuite la figure associée.
        ax = plot_param_importances(study)
        fig = ax.figure
        ax.set_title("Importance des hyperparamètres (Optuna TPE)", fontsize=12)
        fig.tight_layout()
        _save_or_show(fig, save_path, "optuna_importance.png")
    except Exception as e:
        logger.warning("Impossible de tracer l'importance : %s", e)


# ---------------------------------------------------------------------------
# Optuna — Historique d'optimisation
# ---------------------------------------------------------------------------

def plot_optuna_history(
    study,
    save_path: Optional[Path] = None,
) -> None:
    """Trace l'historique des valeurs objectives à travers les trials Optuna."""
    try:
        from optuna.visualization.matplotlib import plot_optimization_history

        ax = plot_optimization_history(study)
        fig = ax.figure
        ax.set_title("Historique d'optimisation Optuna", fontsize=12)
        fig.tight_layout()
        _save_or_show(fig, save_path, "optuna_history.png")
    except Exception as e:
        logger.warning("Impossible de tracer l'historique Optuna : %s", e)


# ---------------------------------------------------------------------------
# Optuna — Coordonnées parallèles
# ---------------------------------------------------------------------------

def plot_optuna_parallel(
    study,
    save_path: Optional[Path] = None,
) -> None:
    """Trace les coordonnées parallèles des hyperparamètres Optuna."""
    try:
        from optuna.visualization.matplotlib import plot_parallel_coordinate

        ax = plot_parallel_coordinate(study)
        fig = ax.figure
        ax.set_title("Coordonnées parallèles — hyperparamètres Optuna", fontsize=12)
        fig.tight_layout()
        _save_or_show(fig, save_path, "optuna_parallel.png")
    except Exception as e:
        logger.warning("Impossible de tracer les coordonnées parallèles : %s", e)


# ---------------------------------------------------------------------------
# 5. Scatter sharpness vs F1-val
# ---------------------------------------------------------------------------

def plot_sharpness_vs_f1(
    sharpness_data: list[dict],
    save_path: Optional[Path] = None,
    xlabel: str = "Sharpness du minimum",
) -> None:
    """
    Scatter plot : sharpness du minimum vs F1-score sur la validation.
    Prédit : corrélation négative (minima plats → meilleure généralisation).

    Parameters
    ----------
    sharpness_data : liste de dicts avec keys 'sharpness', 'val_f1', 'label'
    xlabel : libellé de l'axe X (permet d'utiliser un proxy différent)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sharpnesses = [d["sharpness"] for d in sharpness_data]
    f1s = [d["val_f1"] for d in sharpness_data]
    labels = [d["label"] for d in sharpness_data]
    colors_mapped = [PALETTE[i % len(PALETTE)] for i in range(len(sharpness_data))]

    ax.scatter(sharpnesses, f1s, c=colors_mapped, s=120, zorder=5,
               edgecolors="white", linewidths=1.5)

    for x, y, lbl in zip(sharpnesses, f1s, labels):
        ax.annotate(lbl, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Ligne de tendance
    if len(sharpnesses) > 2:
        z = np.polyfit(sharpnesses, f1s, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(sharpnesses), max(sharpnesses), 100)
        ax.plot(xs, p(xs), "k--", lw=1.5, alpha=0.5, label="Tendance")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("F1-score macro (validation)", fontsize=12)
    ax.set_title(
        "Relation Sharpness ↔ Généralisation\n"
        "Un minimum plat (sharpness faible) → meilleure généralisation",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_or_show(fig, save_path, "sharpness_vs_f1.png")


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: Optional[Path], filename: str) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fp = save_path / filename
        fig.savefig(fp, bbox_inches="tight")
        logger.info("Figure sauvegardée : %s", fp)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="G10 — Visualisations P02")
    parser.add_argument("--output", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--study", type=str, help="Chemin vers optuna_study.pkl")
    parser.add_argument(
        "--metrics",
        type=str,
        default=str(RESULTS_DIR / "baseline_metrics.json"),
        help="Chemin vers baseline_metrics.json (courbes de convergence)",
    )
    parser.add_argument("--grid-csv", type=str, help="Chemin vers grid_p02_results.csv")
    parser.add_argument(
        "--sharpness-json",
        type=str,
        help="Chemin vers un fichier JSON contenant [{sharpness, val_f1, label}, ...]",
    )
    args = parser.parse_args()

    output = Path(args.output)

    # --- Courbes de convergence depuis baseline_metrics.json ---
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        if "history" in metrics:
            label = metrics_path.stem.replace("_metrics", "").replace("_", " ").title()
            plot_convergence_curves({label: metrics["history"]}, save_path=output)
        else:
            logger.warning("Aucune clé 'history' dans %s — courbes de convergence ignorées", metrics_path)
    else:
        logger.warning("Fichier métriques introuvable : %s", metrics_path)

    # --- Heatmap depuis les résultats de la grille ---
    if args.grid_csv:
        df = pd.read_csv(args.grid_csv)
        plot_generalization_heatmap(df, save_path=output)

    # --- Figures Optuna (importance, historique, coordonnées parallèles) ---
    if args.study:
        study_path = Path(args.study)
        if not study_path.exists():
            logger.warning("Fichier étude Optuna introuvable : %s", study_path)
        else:
            with open(study_path, "rb") as f:
                study = pickle.load(f)
            plot_optuna_importance(study, save_path=output)
            plot_optuna_history(study, save_path=output)
            plot_optuna_parallel(study, save_path=output)

    # --- Scatter sharpness vs F1 ---
    # Priorité 1 : fichier JSON fourni manuellement
    if args.sharpness_json:
        sharpness_path = Path(args.sharpness_json)
        if sharpness_path.exists():
            with open(sharpness_path) as f:
                sharpness_data = json.load(f)
            plot_sharpness_vs_f1(sharpness_data, save_path=output)
        else:
            logger.warning("Fichier sharpness introuvable : %s", sharpness_path)
    else:
        # Priorité 2 : construire automatiquement depuis optuna_trials.csv
        # Proxy : |gap train-val| représente la "sharpness" du minimum
        trials_csv = RESULTS_DIR / "optuna_trials.csv"
        if trials_csv.exists():
            df_trials = pd.read_csv(trials_csv)
            df_trials = df_trials[df_trials["state"] == "COMPLETE"].dropna(
                subset=["value", "user_attrs_train_f1", "user_attrs_gap"]
            )
            if not df_trials.empty:
                sharpness_data = [
                    {
                        "sharpness": abs(row["user_attrs_gap"]),
                        "val_f1": row["value"],
                        "label": (
                            f"t{int(row['number'])} "
                            f"dr={row['params_dropout']:.1f} "
                            f"wd={row['params_weight_decay']:.0e}"
                        ),
                    }
                    for _, row in df_trials.iterrows()
                ]
                plot_sharpness_vs_f1(
                    sharpness_data,
                    save_path=output,
                    xlabel="Proxy sharpness : |F1_train − F1_val|",
                )
            else:
                logger.warning("Aucun trial COMPLETE dans %s — scatter sharpness ignoré", trials_csv)
        else:
            logger.warning("optuna_trials.csv introuvable — scatter sharpness ignoré")

    logger.info("Visualisations terminées. Figures dans : %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
