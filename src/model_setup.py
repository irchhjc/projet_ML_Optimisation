"""
model_setup.py — Initialisation de CamemBERT-base (M04)

Points clés :
  - Dropout configurable depuis l'extérieur (requis par la problématique P02)
  - Adaptation automatique CPU / GPU
  - Option quantification dynamique (réduction mémoire ~4x sur CPU)
"""
from __future__ import annotations

import logging

import torch
from transformers import (
    AutoTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
)

from src.config import (
    LABEL_NAMES,
    MODEL_MAX_LEN,
    MODEL_NAME,
    NUM_LABELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Détection du device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Retourne le meilleur device disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU détecté : %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        # Optimisation multi-threading CPU (16 Go RAM → plus de threads)
        torch.set_num_threads(8)
        logger.info("Exécution sur CPU (%d threads)", torch.get_num_threads())
    return device


# ---------------------------------------------------------------------------
# Chargement du tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Charge le tokenizer CamemBERT."""
    logger.info("Chargement du tokenizer '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=MODEL_MAX_LEN,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Chargement du modèle avec dropout configurable
# ---------------------------------------------------------------------------

def load_model(
    dropout: float = 0.1,
    model_name: str = MODEL_NAME,
    device: torch.device | None = None,
    quantize: bool = False,
) -> tuple[CamembertForSequenceClassification, torch.device]:
    """
    Charge CamemBERT-base pour la classification binaire avec dropout configurable.

    Parameters
    ----------
    dropout : float
        Taux de dropout appliqué à la tête de classification ET aux couches
        d'attention (hidden_dropout_prob, attention_probs_dropout_prob).
        Valeurs testées dans P02 : {0.0, 0.1, 0.3}
    model_name : str
        Identifiant HuggingFace du modèle pré-entraîné.
    device : torch.device | None
        Device cible. Détecté automatiquement si None.
    quantize : bool
        Active la quantification dynamique INT8 (réduit la mémoire ~4x sur CPU).

    Returns
    -------
    model : CamembertForSequenceClassification
    device : torch.device
    """
    if device is None:
        device = get_device()

    logger.info(
        "Chargement de CamemBERT-base | dropout=%.2f | device=%s | quantize=%s",
        dropout, device, quantize,
    )

    # Modification de la config pour injecter le dropout custom
    config = CamembertConfig.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        classifier_dropout=dropout,   # tête de classification
    )

    model = CamembertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,  # si la tête est réinitialisée
    )

    # Quantification dynamique (optionnelle, CPU uniquement)
    if quantize and device.type == "cpu":
        from torch.quantization import quantize_dynamic
        model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("Quantification dynamique INT8 activée.")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Paramètres entraînables : %s M", f"{n_params / 1e6:.1f}")

    return model, device


# ---------------------------------------------------------------------------
# Résumé du modèle
# ---------------------------------------------------------------------------

def model_summary(model: CamembertForSequenceClassification) -> None:
    """Affiche un résumé compact du modèle."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"  Modèle : CamemBERT-base (M04)")
    print(f"  Paramètres totaux    : {total / 1e6:.1f} M")
    print(f"  Paramètres entraîn.  : {trainable / 1e6:.1f} M")
    print(f"  Hidden dropout       : {model.config.hidden_dropout_prob}")
    print(f"  Attention dropout    : {model.config.attention_probs_dropout_prob}")
    print(f"  Classifier dropout   : {model.config.classifier_dropout}")
    print(f"{'='*50}\n")
