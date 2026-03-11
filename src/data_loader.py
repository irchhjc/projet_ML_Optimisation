"""
data_loader.py — Chargement et préparation du dataset Allociné (D05)

Stratégie CPU :
  - Sous-échantillonnage équilibré par classe
  - Tokenisation avec troncature à MAX_SEQ_LENGTH
  - Mise en cache locale pour éviter les re-téléchargements
"""
from __future__ import annotations

import json
import logging
import random
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

import src.config as _cfg
from src.config import (
    DATASET_NAME,
    DATA_DIR,
    LABEL_COLUMN,
    NUM_LABELS,
    TEXT_COLUMN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chargement brut
# ---------------------------------------------------------------------------

# Cache global en mémoire pour éviter de recharger le dataset plusieurs fois
_RAW_DATASET: Optional[DatasetDict] = None

def load_raw_dataset() -> DatasetDict:
    """
    Charge le dataset Allociné depuis HuggingFace Hub.
    En cas d'absence de connexion, lève une erreur explicite.
    """
    global _RAW_DATASET

    # Si déjà chargé dans ce processus, on réutilise le cache mémoire
    if _RAW_DATASET is not None:
        return _RAW_DATASET

    cache_path = DATA_DIR / f"{DATASET_NAME}_raw.json"

    # 1) Tentative de chargement depuis le cache JSON local
    if cache_path.exists():
        logger.info(
            "Chargement du dataset '%s' depuis le cache JSON local : %s",
            DATASET_NAME,
            cache_path,
        )
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            splits = {
                split_name: Dataset.from_list(examples)
                for split_name, examples in data.items()
            }
            dataset = DatasetDict(splits)
            _RAW_DATASET = dataset
            return dataset
        except Exception as exc:
            logger.warning(
                "Échec du chargement du cache JSON (%s), recours à load_dataset : %s",
                cache_path,
                exc,
            )

    # 2) Fallback : chargement depuis HuggingFace Hub (utilise aussi le cache disque HF)
    logger.info("Chargement du dataset '%s' depuis HuggingFace Hub...", DATASET_NAME)
    try:
        dataset = load_dataset(DATASET_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger '{DATASET_NAME}'. "
            "Vérifiez votre connexion ou pré-téléchargez le dataset."
        ) from exc

    logger.info("Dataset chargé : %s", dataset)

    # Sauvegarde légère en JSON pour les prochains runs (texte + label uniquement)
    try:
        export = {}
        for split_name, split in dataset.items():
            export[split_name] = [
                {
                    TEXT_COLUMN: ex[TEXT_COLUMN],
                    LABEL_COLUMN: ex[LABEL_COLUMN],
                }
                for ex in split
            ]
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False)
        logger.info("Cache JSON sauvegardé : %s", cache_path)
    except Exception as exc:
        logger.warning("Impossible de sauvegarder le cache JSON (%s) : %s", cache_path, exc)

    # Mise en cache mémoire (HuggingFace gère déjà le cache disque)
    _RAW_DATASET = dataset
    return dataset


# ---------------------------------------------------------------------------
# Sous-échantillonnage équilibré
# ---------------------------------------------------------------------------

def balanced_subsample(
    dataset,
    n_per_class: int,
    num_labels: int = NUM_LABELS,
    seed: int = 42,
) -> list[dict]:
    """
    Retourne n_per_class exemples par classe (sous-ensemble équilibré).

    Parameters
    ----------
    dataset : HuggingFace Dataset (split unique)
    n_per_class : Nombre d'exemples souhaités par classe
    num_labels : Nombre de classes
    seed : Graine aléatoire pour la reproductibilité
    """
    random.seed(seed)
    np.random.seed(seed)

    examples_by_class: dict[int, list] = {k: [] for k in range(num_labels)}
    for ex in dataset:
        examples_by_class[ex[LABEL_COLUMN]].append(ex)

    subset = []
    for label, examples in examples_by_class.items():
        if len(examples) < n_per_class:
            logger.warning(
                "Classe %d : seulement %d exemples disponibles (demandé %d).",
                label, len(examples), n_per_class,
            )
            selected = examples
        else:
            selected = random.sample(examples, n_per_class)
        subset.extend(selected)

    random.shuffle(subset)
    return subset


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_dataset(
    examples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> dict:
    """
    Tokenise une liste d'exemples avec padding et troncature.
    Retourne un dict compatible avec HuggingFace Trainer.
    """
    if max_length is None:
        max_length = _cfg.MAX_SEQ_LENGTH

    texts = [ex[TEXT_COLUMN] for ex in examples]
    labels = [ex[LABEL_COLUMN] for ex in examples]

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encodings["labels"] = labels
    return encodings


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset


class AllocinéDataset(Dataset):
    """Dataset PyTorch pour Allociné tokenisé."""

    def __init__(self, encodings: dict) -> None:
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(encodings["labels"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    seed: int = 42,
) -> tuple[AllocinéDataset, AllocinéDataset, AllocinéDataset]:
    """
    Pipeline complet :
      1. Charge Allociné
      2. Sous-échantillonne chaque split
      3. Tokenise
      4. Retourne (train, val, test) datasets PyTorch

    Returns
    -------
    train_ds, val_ds, test_ds : AllocinéDataset
    """
    if n_train is None:
        n_train = _cfg.N_TRAIN_PER_CLASS
    if n_val is None:
        n_val = _cfg.N_VAL_PER_CLASS
    if n_test is None:
        n_test = _cfg.N_TEST_PER_CLASS

    raw = load_raw_dataset()

    logger.info("Sous-échantillonnage : %d exemples/classe (train), %d (val), %d (test)",
                n_train, n_val, n_test)

    train_samples = balanced_subsample(raw["train"], n_train, seed=seed)
    val_samples = balanced_subsample(raw["validation"], n_val, seed=seed)
    test_samples = balanced_subsample(raw["test"], n_test, seed=seed)

    logger.info("Taille finale — train: %d | val: %d | test: %d",
                len(train_samples), len(val_samples), len(test_samples))

    train_enc = tokenize_dataset(train_samples, tokenizer)
    val_enc = tokenize_dataset(val_samples, tokenizer)
    test_enc = tokenize_dataset(test_samples, tokenizer)

    return AllocinéDataset(train_enc), AllocinéDataset(val_enc), AllocinéDataset(test_enc)


# ---------------------------------------------------------------------------
# Analyse exploratoire rapide
# ---------------------------------------------------------------------------

def dataset_stats(dataset: DatasetDict) -> None:
    """Affiche des statistiques descriptives basiques sur le dataset brut."""
    for split_name, split in dataset.items():
        label_counts = {}
        lengths = []
        for ex in split:
            label_counts[ex[LABEL_COLUMN]] = label_counts.get(ex[LABEL_COLUMN], 0) + 1
            lengths.append(len(ex[TEXT_COLUMN].split()))

        print(f"\n── {split_name} ({len(split)} exemples) ──")
        print(f"  Distribution des classes : {label_counts}")
        print(f"  Longueur moyenne (tokens approx.) : {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
        print(f"  Longueur max : {max(lengths)} | min : {min(lengths)}")
