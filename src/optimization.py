"""
optimization.py — Étude Optuna pour la problématique P02

Recherche Bayésienne sur :
  - weight_decay ∈ {1e-5, 1e-4, 1e-3, 1e-2}
  - dropout      ∈ {0.0, 0.1, 0.3}
  - learning_rate ∈ [1e-6, 5e-4] (log-scale)

Métriques rapportées : F1-score macro (val), gap de généralisation
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

import optuna
import pandas as pd

from src.config import (
    BaselineConfig,
    LandscapeConfig,
    OPTUNA_DB,
    OPTUNA_DIRECTION,
    OPTUNA_N_TRIALS,
    OPTUNA_STUDY_NAME,
    RESULTS_DIR,
    SearchSpace,
)
from src.data_loader import prepare_datasets
from src.metrics import compute_metrics, generalization_gap, format_trial_summary
from src.model_setup import get_device, load_model, load_tokenizer
from src.trainer import CamembertTrainer, TrainConfig

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Objective Optuna
# ---------------------------------------------------------------------------

def build_objective(
    tokenizer,
    device,
    search_space: SearchSpace,
    n_train: int = 500,
    n_val: int = 150,
    seed: int = 42,
):
    """
    Fabrique la fonction objectif pour Optuna.

    L'utilisation d'une closure permet de pré-charger les données
    une seule fois pour tous les trials.
    """
    # Chargement des données une fois (tokenizer identique pour tous les trials)
    logger.info("Préparation des datasets pour l'optimisation...")
    train_ds, val_ds, _ = prepare_datasets(
        tokenizer,
        n_train=n_train // 2,   # n_per_class
        n_val=n_val // 2,
        seed=seed,
    )
    logger.info("Datasets prêts : %d train | %d val", len(train_ds), len(val_ds))

    def objective(trial: optuna.Trial) -> float:
        """
        Un trial = une combinaison (weight_decay, dropout, lr).
        Retourne le F1-score macro sur la validation.
        """
        # ── Échantillonnage des hyperparamètres ──────────────────────────
        weight_decay = trial.suggest_categorical(
            "weight_decay", search_space.weight_decay_choices
        )
        dropout = trial.suggest_categorical(
            "dropout", search_space.dropout_choices
        )
        lr = trial.suggest_float(
            "learning_rate",
            search_space.lr_low,
            search_space.lr_high,
            log=True,
        )

        logger.info(
            "\n[Trial #%03d] weight_decay=%.1e | dropout=%.2f | lr=%.2e",
            trial.number, weight_decay, dropout, lr,
        )

        # ── Chargement du modèle avec le dropout du trial ────────────────
        model, _ = load_model(dropout=dropout, device=device)

        # ── Configuration de l'entraînement (réduite pour Optuna) ────────
        config = TrainConfig(
            learning_rate=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            batch_size=16,
            gradient_accumulation_steps=2,
            num_epochs=2,                  # Réduit pour accélérer la recherche
            warmup_ratio=0.1,
            early_stopping_patience=2,
            seed=seed + trial.number,      # Seed différente par trial
        )

        trainer = CamembertTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
            device=device,
        )

        result = trainer.train()

        # Métriques complémentaires stockées dans le trial (pour analyse)
        history = result["history"]
        if history["train_f1"]:
            best_train_f1 = max(history["train_f1"])
            gap = generalization_gap(best_train_f1, result["best_val_f1"])
            trial.set_user_attr("train_f1", best_train_f1)
            trial.set_user_attr("gap", gap["gap"])
            trial.set_user_attr("gap_pct", gap["gap_pct"])

        trial.set_user_attr("total_time_s", result["total_time_s"])

        # Pruning si performance insuffisante (optimisation CPU)
        if result["best_val_f1"] < 0.55:
            raise optuna.TrialPruned()

        return result["best_val_f1"]

    return objective, train_ds, val_ds


# ---------------------------------------------------------------------------
# Lancement de l'étude
# ---------------------------------------------------------------------------

def run_study(
    n_trials: int = OPTUNA_N_TRIALS,
    output_dir: Path = RESULTS_DIR,
    seed: int = 42,
) -> optuna.Study:
    """
    Lance l'étude Optuna et sauvegarde les résultats.

    L'algorithme TPE (Tree-structured Parzen Estimator) est utilisé
    par défaut dans Optuna — c'est une méthode Bayésienne.
    """
    search_space = SearchSpace()
    device = get_device()
    tokenizer = load_tokenizer()

    objective, train_ds, val_ds = build_objective(
        tokenizer=tokenizer,
        device=device,
        search_space=search_space,
        seed=seed,
    )

    # Création / reprise de l'étude
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3)

    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction=OPTUNA_DIRECTION,
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
    )

    logger.info(
        "Démarrage de l'étude Optuna — %d trials | %d existants",
        n_trials, len(study.trials),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[_logging_callback],
    )

    # ── Sauvegarde des résultats ─────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pickle de l'étude complète
    with open(output_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    # CSV des trials
    df = study.trials_dataframe()
    df.to_csv(output_dir / "optuna_trials.csv", index=False)

    # Meilleurs paramètres en JSON
    best_params = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # ── Rapport console ──────────────────────────────────────────────────
    _print_study_summary(study)

    return study


def _logging_callback(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
    """Callback appelé après chaque trial pour afficher la progression."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        logger.info(
            "✓ Trial #%03d terminé — F1_val=%.4f | best=%.4f",
            trial.number, trial.value, study.best_value,
        )


def _print_study_summary(study: optuna.Study) -> None:
    """Affiche un résumé de l'étude Optuna."""
    print("\n" + "═" * 60)
    print("  RÉSUMÉ DE L'ÉTUDE OPTUNA — G10 (P02)")
    print("═" * 60)
    print(f"  Trials complétés : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Meilleur F1-val  : {study.best_value:.4f}")
    print(f"  Meilleurs params :")
    for k, v in study.best_params.items():
        print(f"    {k:25s} = {v}")
    print("═" * 60)

    # Top 5 trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print("\n  Top 5 trials :")
    print(f"  {'#':>4}  {'weight_decay':>14}  {'dropout':>8}  {'lr':>10}  {'F1_val':>8}  {'gap':>8}")
    print("  " + "─" * 58)
    for t in top5:
        print(
            f"  {t.number:>4}  "
            f"{t.params.get('weight_decay', 0):>14.1e}  "
            f"{t.params.get('dropout', 0):>8.2f}  "
            f"{t.params.get('learning_rate', 0):>10.2e}  "
            f"{t.value:>8.4f}  "
            f"{t.user_attrs.get('gap', float('nan')):>8.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Baseline complète (O1 / O3 / O6)
# ---------------------------------------------------------------------------

def run_baseline(
    output_dir: Path = RESULTS_DIR,
    seed: int = 42,
) -> dict:
    """Entraîne une configuration baseline complète et évalue sur test.

    Aligne le code avec les objectifs :
      - O1 : baseline F1 train/val/test
      - O3 : gaps de généralisation train-val et train-test
    """
    device = get_device()
    tokenizer = load_tokenizer()

    # Dataset complet (taille CPU-safe définie dans config)
    train_ds, val_ds, test_ds = prepare_datasets(tokenizer, seed=seed)

    baseline = BaselineConfig()
    model, _ = load_model(dropout=baseline.dropout, device=device)

    train_cfg = TrainConfig(
        learning_rate=baseline.learning_rate,
        weight_decay=baseline.weight_decay,
        dropout=baseline.dropout,
        batch_size=baseline.batch_size,
        gradient_accumulation_steps=baseline.gradient_accumulation_steps,
        num_epochs=baseline.num_epochs,
        warmup_ratio=baseline.warmup_ratio,
        max_steps=baseline.max_steps,
        early_stopping_patience=2,
        seed=baseline.seed,
    )

    trainer = CamembertTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=train_cfg,
        device=device,
    )

    train_result = trainer.train()
    test_metrics = trainer.evaluate_on_test(test_ds)

    best_train_f1 = train_result.get("best_train_f1", 0.0)
    best_val_f1 = train_result["best_val_f1"]

    gap_train_val = generalization_gap(best_train_f1, best_val_f1)
    gap_train_test = generalization_gap(best_train_f1, test_metrics["f1_macro"])

    summary = {
        "baseline_hparams": baseline.__dict__,
        "best_train_f1": best_train_f1,
        "best_val_f1": best_val_f1,
        "f1_test": test_metrics["f1_macro"],
        "gap_train_val": gap_train_val,
        "gap_train_test": gap_train_test,
        "test_metrics": test_metrics,
        "history": train_result["history"],
        "total_time_s": train_result["total_time_s"],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Baseline sauvegardée dans : %s", output_dir / "baseline_metrics.json")

    return summary


# ---------------------------------------------------------------------------
# Grille exhaustive P02 (pour visualisation heatmap)
# ---------------------------------------------------------------------------

def run_grid_search_p02(
    tokenizer,
    device,
    output_dir: Path = RESULTS_DIR,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Évalue toutes les 12 combinaisons (weight_decay × dropout) du protocole P02
    avec un learning rate fixe pour isoler l'effet de la régularisation.

    Cette fonction complète l'étude Optuna par une analyse déterministe.
    """
    search_space = SearchSpace()
    results = []

    train_ds, val_ds, _ = prepare_datasets(
        tokenizer,
        n_train=250,   # n_per_class
        n_val=100,
        seed=seed,
    )

    lr_fixed = 2e-5  # Fixé pour isoler weight_decay et dropout

    for wd in search_space.weight_decay_choices:
        for dp in search_space.dropout_choices:
            logger.info("Grid — weight_decay=%.1e | dropout=%.2f", wd, dp)

            model, _ = load_model(dropout=dp, device=device)
            config = TrainConfig(
                learning_rate=lr_fixed,
                weight_decay=wd,
                dropout=dp,
                num_epochs=2,
                early_stopping_patience=2,
                seed=seed,
            )
            trainer = CamembertTrainer(model, train_ds, val_ds, config, device)
            result = trainer.train()

            history = result["history"]
            train_f1 = max(history["train_f1"]) if history["train_f1"] else 0.0
            val_f1 = result["best_val_f1"]

            results.append({
                "weight_decay": wd,
                "dropout": dp,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "gap": train_f1 - val_f1,
                "time_s": result["total_time_s"],
            })

            logger.info("  → F1_train=%.4f | F1_val=%.4f | gap=%.4f",
                        train_f1, val_f1, train_f1 - val_f1)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "grid_p02_results.csv", index=False)
    logger.info("Résultats grille sauvegardés : %s", output_dir / "grid_p02_results.csv")
    return df


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="G10 — Optimisation P02 (Optuna)")
    parser.add_argument("--n-trials", type=int, default=OPTUNA_N_TRIALS)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["baseline", "optuna", "grid", "both"],
        default="optuna",
        help="Mode : baseline simple ou étude optuna / grille",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    device = get_device()
    tokenizer = load_tokenizer()

    if args.mode == "baseline":
        run_baseline(output_dir=output_dir, seed=args.seed)

    if args.mode in ("optuna", "both"):
        run_study(n_trials=args.n_trials, output_dir=output_dir, seed=args.seed)

    if args.mode in ("grid", "both"):
        run_grid_search_p02(tokenizer=tokenizer, device=device,
                            output_dir=output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
