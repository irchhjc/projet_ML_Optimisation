"""loss_landscape_analysis.py — Analyse approfondie du loss landscape (P02)

Objectif
--------
Comparer la géométrie du minimum (plat vs pointu) pour plusieurs
configurations de régularisation sur Allociné :
  - configuration sous‑régularisée (peu de weight decay / dropout)
  - configuration baseline (config par défaut du projet)
  - configuration fortement régularisée (weight decay / dropout élevés)
  - éventuellement : meilleure configuration trouvée par Optuna

Pour chaque configuration, on :
  1. entraîne rapidement un modèle (quelques époques, dataset réduit)
  2. calcule un loss landscape 1D autour du minimum trouvé
  3. trace une figure de comparaison avec la sharpness pour chaque courbe

Lance avec, depuis la racine du projet :

    poetry run run-landscape-loss

Les figures seront sauvegardées dans results/figures.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import (
    BaselineConfig,
    FIGURES_DIR,
    RESULTS_DIR,
    LandscapeConfig,
)
from src.data_loader import prepare_datasets
from src.model_setup import get_device, load_model, load_tokenizer
from src.trainer import CamembertTrainer, TrainConfig
from src.visualization import compute_loss_landscape_1d, plot_loss_landscape_comparison


# ---------------------------------------------------------------------------
# Version allégée pour CPU (Section 6.1 du rapport)
# ---------------------------------------------------------------------------

def evaluate_on_subset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    n_samples: int = 50,
) -> float:
    """Évalue la loss moyenne sur un sous-ensemble de `n_samples` exemples."""
    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=16, shuffle=False)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def compute_loss_landscape_light(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    n_points: int = 8,
    epsilon: float = 0.05,
) -> tuple[np.ndarray, list[float]]:
    """
    Version légère du loss landscape 1D pour CPU.

    Différences avec compute_loss_landscape_1d :
      - Normalisation globale (une seule norme sur tous les paramètres)
        au lieu de la filter normalization (Li et al. 2018)
      - Évaluation sur seulement 50 exemples (evaluate_on_subset)
      - Grille réduite à n_points=8 par défaut

    Algorithme :
      1. Sauvegarder θ*
      2. Direction aléatoire normalisée globalement
      3. Évaluer L(θ* + α·d) pour α ∈ [-ε, +ε]
      4. Restaurer θ*
    """
    model.eval()
    original_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire normalisée (normalisation globale)
    direction = [torch.randn_like(p) for p in original_params]
    total_norm = sum(d.norm().item() for d in direction)
    if total_norm > 0:
        direction = [d / total_norm for d in direction]

    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        # Appliquer la perturbation
        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + float(alpha) * d.to(p0.device)

        loss = evaluate_on_subset(model, dataset, device, n_samples=50)
        losses.append(loss)

    # Restaurer θ*
    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0.to(p.device)

    return alphas, losses

logger = logging.getLogger(__name__)


@dataclass
class LandscapeRunConfig:
    label: str
    weight_decay: float
    dropout: float
    learning_rate: float


def _load_best_optuna_config(results_dir: Path) -> LandscapeRunConfig | None:
    """Charge la meilleure configuration trouvée par Optuna si disponible."""
    best_path = results_dir / "best_params.json"
    if not best_path.exists():
        logger.info("Aucun best_params.json trouvé, pas de config Optuna ajoutée.")
        return None

    with open(best_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data.get("best_params", {})
    if not {"weight_decay", "dropout", "learning_rate"}.issubset(best.keys()):
        logger.warning("best_params.json ne contient pas les clés complètes attendues.")
        return None

    label = (
        f"Optuna best (wd={best['weight_decay']:.1e}, "
        f"dropout={best['dropout']:.2f})"
    )
    return LandscapeRunConfig(
        label=label,
        weight_decay=float(best["weight_decay"]),
        dropout=float(best["dropout"]),
        learning_rate=float(best["learning_rate"]),
    )


def _train_and_compute_landscape(
    cfg: LandscapeRunConfig,
    tokenizer,
    device: torch.device,
    train_ds,
    val_ds,
    landscape_cfg: LandscapeConfig,
) -> Tuple[str, Tuple]:
    """Entraîne un modèle pour une config donnée et calcule son loss landscape 1D."""
    logger.info(
        "\n=== Configuration '%s' ===\n  weight_decay = %.1e | dropout = %.2f | lr = %.2e",
        cfg.label, cfg.weight_decay, cfg.dropout, cfg.learning_rate,
    )

    model, _ = load_model(dropout=cfg.dropout, device=device)

    train_config = TrainConfig(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        dropout=cfg.dropout,
        batch_size=16,
        gradient_accumulation_steps=2,
        num_epochs=2,  # entraînement court pour l'analyse
        warmup_ratio=0.1,
        early_stopping_patience=2,
        seed=42,
    )

    trainer = CamembertTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=train_config,
        device=device,
    )

    result = trainer.train()
    logger.info("F1-val (config '%s') = %.4f", cfg.label, result["best_val_f1"])

    # Sur CPU : version légère (normalisation globale, 50 samples)
    # Sur GPU : version complète avec filter normalization (Li et al. 2018)
    if device.type == "cpu":
        logger.info("CPU détecté → compute_loss_landscape_light (n_points=8, n_samples=50)")
        alphas, losses = compute_loss_landscape_light(
            model=model,
            dataset=val_ds,
            device=device,
            n_points=landscape_cfg.n_points,
            epsilon=landscape_cfg.epsilon,
        )
    else:
        alphas, losses = compute_loss_landscape_1d(
            model=model,
            dataset=val_ds,
            device=device,
            cfg=landscape_cfg,
        )
    return cfg.label, (alphas, losses), float(result["best_val_f1"])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    device = get_device()
    tokenizer = load_tokenizer()

    # Dataset adapté à 16 Go RAM / CPU
    train_ds, val_ds, _ = prepare_datasets(
        tokenizer,
        n_train=350,  # par classe (16 Go RAM)
        n_val=150,
        n_test=100,
        seed=42,
    )

    baseline = BaselineConfig()

    configs = [
        # Sous-régularisation : gap de généralisation attendu élevé
        LandscapeRunConfig(
            label="Sous-régularisé (wd=1e-5, dropout=0.0)",
            weight_decay=1e-5,
            dropout=0.0,
            learning_rate=baseline.learning_rate,
        ),
        # Baseline du projet
        LandscapeRunConfig(
            label="Baseline (wd=1e-4, dropout=0.1)",
            weight_decay=baseline.weight_decay,
            dropout=baseline.dropout,
            learning_rate=baseline.learning_rate,
        ),
        # Fortement régularisé : minima attendus plus plats mais F1 plus faible
        LandscapeRunConfig(
            label="Fortement régularisé (wd=1e-2, dropout=0.3)",
            weight_decay=1e-2,
            dropout=0.3,
            learning_rate=baseline.learning_rate,
        ),
    ]

    # Ajout éventuel de la meilleure config Optuna
    optuna_cfg = _load_best_optuna_config(RESULTS_DIR)
    if optuna_cfg is not None:
        configs.append(optuna_cfg)

    landscape_cfg = LandscapeConfig()

    results: Dict[str, Tuple] = {}
    val_f1s: Dict[str, float] = {}
    for cfg in configs:
        label, curve, val_f1 = _train_and_compute_landscape(
            cfg,
            tokenizer=tokenizer,
            device=device,
            train_ds=train_ds,
            val_ds=val_ds,
            landscape_cfg=landscape_cfg,
        )
        results[label] = curve
        val_f1s[label] = val_f1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sharpnesses = plot_loss_landscape_comparison(results, save_path=FIGURES_DIR)

    logger.info("Analyse du loss landscape terminée.")
    logger.info("Sharpness par configuration :")
    for label, sharp in sharpnesses.items():
        logger.info("  %-40s  %.6f", label, sharp)

    # Sauvegarde pour plot_sharpness_vs_f1 dans run_pipeline.py
    import json
    sharpness_list = [
        {"label": label, "sharpness": sharp, "val_f1": val_f1s.get(label, 0.0)}
        for label, sharp in sharpnesses.items()
    ]
    sharpness_path = FIGURES_DIR / "sharpness_data.json"
    with open(sharpness_path, "w", encoding="utf-8") as fh:
        json.dump(sharpness_list, fh, indent=2, ensure_ascii=False)
    logger.info("sharpness_data.json sauvegardé : %s", sharpness_path)


if __name__ == "__main__":
    main()
