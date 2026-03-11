"""Run the full G10 CamemBERT * Allociné pipeline.

Steps (by default):
  1. Baseline training + test evaluation
  2. Optuna study (P02: weight_decay * dropout * lr)
  3. Grid search P02 (weight_decay * dropout)
  4. Visualisations (heatmap, Optuna importances, convergence curves, sharpness scatter)
  5. Loss landscape 1D analysis

Usage (from project root, with Poetry env active):

    poetry run python run_pipeline.py

You can selectively disable steps with flags, e.g.:

    poetry run python run_pipeline.py --no-optuna --no-grid

"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import FIGURES_DIR, RESULTS_DIR, OPTUNA_N_TRIALS
from src.loss_landscape_analysis import main as run_loss_landscape
from src.optimization import run_baseline, run_grid_search_p02, run_study
from src.visualization import (
    plot_generalization_heatmap,
    plot_optuna_importance,
    plot_convergence_curves,
    plot_sharpness_vs_f1,
)


logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    """Fixe toutes les graines pour la reproductibilité (critère protocole 15%)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_visualizations(results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    """Generate key visualizations from Optuna and grid results if available."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)

    study_path = results_dir / "optuna_study.pkl"
    grid_csv = results_dir / "grid_p02_results.csv"
    baseline_json = results_dir / "baseline_metrics.json"

    # ── Heatmap gap train/val, gap train/test, F1-val ────────────────────
    if grid_csv.exists():
        logger.info("Plotting generalization heatmap from %s", grid_csv)
        df = pd.read_csv(grid_csv)
        plot_generalization_heatmap(df, save_path=figures_dir)
    else:
        logger.info("No grid_p02_results.csv found in %s, skipping heatmap.", results_dir)

    # ── Importance des hyperparamètres Optuna ────────────────────────────
    if study_path.exists():
        logger.info("Plotting Optuna parameter importance from %s", study_path)
        with study_path.open("rb") as f:
            study = pickle.load(f)
        plot_optuna_importance(study, save_path=figures_dir)
    else:
        logger.info("No optuna_study.pkl found in %s, skipping Optuna importance plot.", results_dir)

    # ── Courbes de convergence baseline vs best Optuna ───────────────────
    # (critère Visualisations 10% + Amélioration 15%)
    if baseline_json.exists():
        with baseline_json.open("r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        histories = {"Baseline": baseline_data["history"]}
        plot_convergence_curves(histories, save_path=figures_dir)
        logger.info("Courbes de convergence sauvegardées.")

    # ── Scatter sharpness vs F1-val ──────────────────────────────────────
    # (critère Interprétation 10% : relier platitude ↔ généralisation)
    sharpness_json = figures_dir / "sharpness_data.json"
    if sharpness_json.exists():
        with sharpness_json.open("r", encoding="utf-8") as f:
            sharpness_data = json.load(f)
        plot_sharpness_vs_f1(sharpness_data, save_path=figures_dir)
        logger.info("Scatter sharpness vs F1 sauvegardé.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full G10 CamemBERT P02 pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-trials", type=int, default=OPTUNA_N_TRIALS, help="Optuna trials")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline run")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna study")
    parser.add_argument("--no-grid", action="store_true", help="Skip grid search P02")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--no-landscape", action="store_true", help="Skip loss landscape analysis")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting full pipeline with seed=%d", args.seed)
    _set_global_seed(args.seed)

    # Construire la liste des étapes activées
    steps: list[tuple[str, callable]] = []

    if args.no_baseline:
        logger.info("Skipping baseline step (requested by flag).")
    else:
        steps.append((
            "Baseline training + test evaluation",
            lambda: run_baseline(output_dir=RESULTS_DIR, seed=args.seed),
        ))

    if args.no_optuna:
        logger.info("Skipping Optuna study (requested by flag).")
    else:
        steps.append((
            f"Optuna study (n_trials={args.n_trials})",
            lambda: run_study(n_trials=args.n_trials, output_dir=RESULTS_DIR, seed=args.seed),
        ))

    if args.no_grid:
        logger.info("Skipping grid search (requested by flag).")
    else:
        def _run_grid() -> None:
            logger.info("Running grid search P02 (weight_decay * dropout)...")
            from src.model_setup import get_device, load_tokenizer

            device = get_device()
            tokenizer = load_tokenizer()
            run_grid_search_p02(
                tokenizer=tokenizer,
                device=device,
                output_dir=RESULTS_DIR,
                seed=args.seed,
            )

        steps.append(("Grid search P02", _run_grid))

    if args.no_viz:
        logger.info("Skipping visualization step (requested by flag).")
    else:
        steps.append((
            "Visualizations (heatmap, Optuna importance)",
            lambda: run_visualizations(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR),
        ))

    if args.no_landscape:
        logger.info("Skipping loss landscape analysis (requested by flag).")
    else:
        steps.append(("Loss landscape 1D analysis", run_loss_landscape))

    if not steps:
        logger.info("No pipeline steps enabled; nothing to run.")
        return

    total = len(steps)
    with tqdm(total=total, desc="Pipeline", unit="step") as pbar:
        for idx, (label, fn) in enumerate(steps, start=1):
            logger.info("[%d/%d] %s...", idx, total, label)
            fn()
            pbar.update(1)

    logger.info("Pipeline completed. Results in %s and figures in %s", RESULTS_DIR, FIGURES_DIR)

    # ── Rapport de gain : Amélioration vs baseline (critère 15%) ────────
    baseline_json = RESULTS_DIR / "baseline_metrics.json"
    best_json = RESULTS_DIR / "best_params.json"
    if baseline_json.exists() and best_json.exists():
        with baseline_json.open() as f:
            b = json.load(f)
        with best_json.open() as f:
            p = json.load(f)
        baseline_f1 = b.get("best_val_f1", 0.0)
        best_f1 = p.get("best_value", 0.0)
        gain = best_f1 - baseline_f1
        gain_pct = (gain / baseline_f1 * 100) if baseline_f1 > 0 else 0.0
        logger.info(
            "\n%s\n  Gain Optuna vs Baseline\n  F1_baseline = %.4f | F1_best = %.4f | gain = %+.4f (%+.1f%%)\n%s",
            "=" * 55, baseline_f1, best_f1, gain, gain_pct, "=" * 55,
        )
        # Sauvegarde du gain pour le rapport
        gain_summary = {
            "baseline_val_f1": baseline_f1,
            "best_optuna_val_f1": best_f1,
            "gain_absolute": gain,
            "gain_pct": gain_pct,
            "baseline_test_f1": b.get("f1_test", None),
        }
        with open(RESULTS_DIR / "improvement_summary.json", "w") as f:
            json.dump(gain_summary, f, indent=2)
        logger.info("Gain sauvegardé dans results/improvement_summary.json")


if __name__ == "__main__":
    main()
