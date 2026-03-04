"""
data_loader.py — Chargement et préparation du dataset Allociné (D05)

Stratégie CPU :
  - Sous-échantillonnage équilibré par classe
  - Tokenisation avec troncature à MAX_SEQ_LENGTH
  - Mise en cache locale pour éviter les re-téléchargements
"""
from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.config import (
    DATASET_NAME,
    LABEL_COLUMN,
    MAX_SEQ_LENGTH,
    N_TEST_PER_CLASS,
    N_TRAIN_PER_CLASS,
    N_VAL_PER_CLASS,
    NUM_LABELS,
    TEXT_COLUMN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chargement brut
# ---------------------------------------------------------------------------

def load_raw_dataset() -> DatasetDict:
    """
    Charge le dataset Allociné depuis HuggingFace Hub.
    En cas d'absence de connexion, lève une erreur explicite.
    """
    logger.info("Chargement du dataset '%s'...", DATASET_NAME)
    try:
        dataset = load_dataset(DATASET_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger '{DATASET_NAME}'. "
            "Vérifiez votre connexion ou pré-téléchargez le dataset."
        ) from exc

    logger.info("Dataset chargé : %s", dataset)
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
    max_length: int = MAX_SEQ_LENGTH,
) -> dict:
    """
    Tokenise une liste d'exemples avec padding et troncature.
    Retourne un dict compatible avec HuggingFace Trainer.
    """
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
    n_train: int = N_TRAIN_PER_CLASS,
    n_val: int = N_VAL_PER_CLASS,
    n_test: int = N_TEST_PER_CLASS,
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
