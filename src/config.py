"""
config.py — Configuration centrale du projet G10
Dataset : Allociné (D05) | Modèle : CamemBERT-base (M04)
Problématique : P02 — Régularisation & Généralisation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Dossier pour stocker les données prétraitées / caches légers
DATA_DIR = ROOT_DIR / "data"

for _d in [RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "allocine"          # HuggingFace Hub ID
DATASET_ID = "D05"
TEXT_COLUMN = "review"
LABEL_COLUMN = "label"
NUM_LABELS = 2
LABEL_NAMES = ["négatif", "positif"]

# Sous-ensembles pour CPU (équilibrés par classe)
N_TRAIN_PER_CLASS = 500            # → 2 000 exemples train total
N_VAL_PER_CLASS = 200             # → 600 exemples val total
N_TEST_PER_CLASS = 200             # → 600  exemples test total
MAX_SEQ_LENGTH = 256               # Truncation mémoire-safe


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
MODEL_NAME = "camembert-base"
MODEL_ID = "M04"
MODEL_MAX_LEN = 512


# ---------------------------------------------------------------------------
# Hyperparamètres par défaut (baseline)
# ---------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    dropout: float = 0.1
    batch_size: int = 16
    gradient_accumulation_steps: int = 2   # batch effectif = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_steps: int = -1                    # -1 = utiliser num_epochs
    seed: int = 42


# ---------------------------------------------------------------------------
# Espace de recherche Optuna (P02)
# ---------------------------------------------------------------------------
@dataclass
class SearchSpace:
    """
    Grid imposé par le protocole P02 :
        weight_decay = {1e-5, 1e-4, 1e-3, 1e-2}
        dropout      = {0.0, 0.1, 0.3}
    Optuna explore ensuite cet espace de façon Bayésienne.
    """
    weight_decay_choices: list[float] = field(
        default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2]
    )
    dropout_choices: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.3]
    )
    # Learning rate : plage continue en log-scale
    lr_low: float = 1e-6
    lr_high: float = 5e-4


# ---------------------------------------------------------------------------
# Loss landscape
# ---------------------------------------------------------------------------
@dataclass
class LandscapeConfig:
    n_points: int = 8          # Nombre de points sur la grille 1D
    epsilon: float = 0.05      # Amplitude de la perturbation
    n_samples_eval: int = 50   # Sous-ensemble pour l'évaluation rapide


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = 20
OPTUNA_STUDY_NAME = "g10_p02_regularisation"
OPTUNA_DIRECTION = "maximize"   # On maximise le F1-score macro
OPTUNA_DB = str(RESULTS_DIR / "optuna_study.db") # pour pouvoir utiliser optuna-dashboard