"""
trainer.py — Boucle d'entraînement custom pour CamemBERT (G10)

Conçu pour :
  - Fonctionner sur CPU avec gradient accumulation
  - Mesurer les métriques train ET val à chaque époque (requis pour P02)
  - Implémenter l'early stopping
  - Sauvegarder les checkpoints du meilleur modèle
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import CHECKPOINTS_DIR
from src.metrics import compute_metrics, generalization_gap, format_trial_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration de l'entraînement
# ---------------------------------------------------------------------------

class TrainConfig:
    """Regroupe tous les hyperparamètres de l'entraînement."""

    def __init__(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,        # déjà appliqué dans le modèle
        batch_size: int = 16,
        gradient_accumulation_steps: int = 2,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        max_steps: int = -1,
        early_stopping_patience: int = 2,
        seed: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CamembertTrainer:
    """
    Boucle d'entraînement avec :
      - Gradient accumulation (simulation grands batch sizes sur CPU)
      - Scheduler linéaire avec warmup
      - Early stopping sur F1-score de validation
      - Logging epoch-level des métriques train et val
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        config: TrainConfig,
        device: torch.device,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.save_dir = save_dir or CHECKPOINTS_DIR

        # Historique pour visualisation
        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "train_acc": [], "val_acc": [],
        }

        self._setup_seed()

    def _setup_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,   # CPU-safe
            pin_memory=False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )
        return train_loader, val_loader

    def _build_optimizer_and_scheduler(
        self, train_loader: DataLoader
    ) -> tuple[AdamW, object]:
        """
        AdamW avec weight_decay configuré.
        Les paramètres de biais et LayerNorm sont exclus du weight decay
        (pratique standard pour les Transformers).
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

        # Calcul du nombre de steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(
            "Optimizer : AdamW | lr=%.2e | weight_decay=%.2e | "
            "total_steps=%d | warmup_steps=%d",
            self.config.learning_rate, self.config.weight_decay,
            total_steps, warmup_steps,
        )

        return optimizer, scheduler

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: AdamW,
        scheduler,
    ) -> tuple[float, dict]:
        """Exécute une époque d'entraînement. Retourne (loss_moy, métriques)."""
        self.model.train()
        total_loss = 0.0
        all_labels, all_preds = [], []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            loss.backward()
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        """Exécute une passe d'évaluation. Retourne (loss_moy, métriques)."""
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())

        avg_loss = total_loss / len(loader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics

    def train(self) -> dict:
        """
        Lance l'entraînement complet avec early stopping.

        Returns
        -------
        dict avec les meilleurs métriques val et l'historique.
        """
        train_loader, val_loader = self._build_dataloaders()
        optimizer, scheduler = self._build_optimizer_and_scheduler(train_loader)

        best_val_f1 = -1.0
        best_model_state = None
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()

            train_loss, train_metrics = self._train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_metrics = self._eval_epoch(val_loader)

            # Historique
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_f1"].append(train_metrics["f1_macro"])
            self.history["val_f1"].append(val_metrics["f1_macro"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            elapsed = time.time() - t0
            gap = generalization_gap(train_metrics["f1_macro"], val_metrics["f1_macro"])

            logger.info(
                "Époque %d/%d — train_loss=%.4f val_loss=%.4f "
                "F1_train=%.4f F1_val=%.4f gap=%.4f (%.1fs)",
                epoch, self.config.num_epochs,
                train_loss, val_loss,
                train_metrics["f1_macro"], val_metrics["f1_macro"],
                gap["gap"], elapsed,
            )

            # Early stopping & checkpoint
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping déclenché à l'époque %d "
                        "(pas d'amélioration depuis %d époques).",
                        epoch, self.config.early_stopping_patience,
                    )
                    break

        # Restaurer le meilleur modèle
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        total_time = time.time() - start_time
        logger.info("Entraînement terminé en %.1f s | Meilleur F1 val : %.4f",
                    total_time, best_val_f1)

        return {
            "best_val_f1": best_val_f1,
            "history": self.history,
            "total_time_s": total_time,
        }

    def evaluate_on_test(self, test_dataset) -> dict:
        """Évalue le modèle sur le jeu de test et affiche le rapport complet."""
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
        )
        test_loss, test_metrics = self._eval_epoch(test_loader)
        logger.info("Test — F1 macro : %.4f | Accuracy : %.4f",
                    test_metrics["f1_macro"], test_metrics["accuracy"])
        return {"test_loss": test_loss, **test_metrics}

    def save_history(self, path: Optional[Path] = None) -> None:
        """Sauvegarde l'historique d'entraînement en JSON."""
        p = path or (self.save_dir / "training_history.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Historique sauvegardé : %s", p)
