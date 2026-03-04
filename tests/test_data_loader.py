"""
tests/test_data_loader.py — Tests unitaires pour data_loader.py
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Tests du sous-échantillonnage
# ---------------------------------------------------------------------------

class FakeDataset:
    """Dataset factice pour les tests (sans téléchargement réseau)."""

    def __init__(self, n_per_class: int = 200, n_classes: int = 2):
        self._data = []
        for label in range(n_classes):
            for i in range(n_per_class):
                self._data.append({
                    "review": f"Critique numéro {i} pour la classe {label}. " * 5,
                    "label": label,
                })

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def test_balanced_subsample_count():
    """Vérifie que balanced_subsample retourne exactement n_per_class × n_classes exemples."""
    from src.data_loader import balanced_subsample

    fake = FakeDataset(n_per_class=200, n_classes=2)
    n_per_class = 50
    result = balanced_subsample(fake, n_per_class=n_per_class, num_labels=2)

    assert len(result) == n_per_class * 2


def test_balanced_subsample_is_balanced():
    """Vérifie que le sous-ensemble est équilibré entre les classes."""
    from src.data_loader import balanced_subsample

    fake = FakeDataset(n_per_class=200, n_classes=2)
    result = balanced_subsample(fake, n_per_class=80, num_labels=2)

    labels = [ex["label"] for ex in result]
    counts = {l: labels.count(l) for l in set(labels)}

    assert len(counts) == 2
    for label, count in counts.items():
        assert count == 80, f"Classe {label} : {count} exemples attendus 80"


def test_balanced_subsample_reproducibility():
    """Vérifie la reproductibilité avec la même graine."""
    from src.data_loader import balanced_subsample

    fake = FakeDataset(n_per_class=200, n_classes=2)
    s1 = balanced_subsample(fake, n_per_class=30, seed=42)
    s2 = balanced_subsample(fake, n_per_class=30, seed=42)

    texts1 = [ex["review"] for ex in s1]
    texts2 = [ex["review"] for ex in s2]
    assert texts1 == texts2, "Les sous-ensembles devraient être identiques avec la même graine"


def test_balanced_subsample_different_seeds():
    """Vérifie que des graines différentes produisent des sous-ensembles différents."""
    from src.data_loader import balanced_subsample

    fake = FakeDataset(n_per_class=200, n_classes=2)
    s1 = balanced_subsample(fake, n_per_class=50, seed=42)
    s2 = balanced_subsample(fake, n_per_class=50, seed=123)

    texts1 = [ex["review"] for ex in s1]
    texts2 = [ex["review"] for ex in s2]
    assert texts1 != texts2, "Des graines différentes devraient produire des ordres différents"


def test_balanced_subsample_insufficient_data():
    """Vérifie le comportement quand il n'y a pas assez d'exemples."""
    from src.data_loader import balanced_subsample

    fake = FakeDataset(n_per_class=10, n_classes=2)
    # Demande plus que disponible → doit prendre tout
    result = balanced_subsample(fake, n_per_class=100, num_labels=2)
    assert len(result) == 20  # Seulement 10×2 disponibles


# ---------------------------------------------------------------------------
# Tests du dataset PyTorch
# ---------------------------------------------------------------------------

def test_allocine_dataset_length():
    """Vérifie que AllocinéDataset retourne la bonne longueur."""
    import torch
    from src.data_loader import AllocinéDataset

    n = 10
    enc = {
        "input_ids": torch.randint(0, 1000, (n, 128)),
        "attention_mask": torch.ones(n, 128, dtype=torch.long),
        "labels": list(range(n)),
    }
    ds = AllocinéDataset(enc)
    assert len(ds) == n


def test_allocine_dataset_item_keys():
    """Vérifie que chaque item contient les bonnes clés."""
    import torch
    from src.data_loader import AllocinéDataset

    n = 5
    enc = {
        "input_ids": torch.randint(0, 1000, (n, 64)),
        "attention_mask": torch.ones(n, 64, dtype=torch.long),
        "labels": [0, 1, 0, 1, 0],
    }
    ds = AllocinéDataset(enc)
    item = ds[0]

    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert isinstance(item["labels"], torch.Tensor)


# ---------------------------------------------------------------------------
# Tests des métriques
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect():
    """Vérifie que les métriques sont 1.0 pour une prédiction parfaite."""
    from src.metrics import compute_metrics

    labels = [0, 1, 0, 1, 0, 1]
    preds  = [0, 1, 0, 1, 0, 1]
    result = compute_metrics(labels, preds)

    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0


def test_generalization_gap():
    """Vérifie le calcul du gap de généralisation."""
    from src.metrics import generalization_gap

    result = generalization_gap(train_f1=0.90, val_f1=0.80)

    assert abs(result["gap"] - 0.10) < 1e-6
    assert abs(result["gap_pct"] - (10 / 90 * 100)) < 1e-4


def test_sharpness_flat_minimum():
    """Un minimum plat devrait avoir une sharpness proche de 0."""
    from src.metrics import compute_sharpness

    base_loss = 0.5
    perturbed = [0.501, 0.499, 0.500, 0.502, 0.498]
    sharpness = compute_sharpness(base_loss, perturbed)

    assert sharpness < 0.01, f"Sharpness trop élevée pour un min plat : {sharpness}"


def test_sharpness_sharp_minimum():
    """Un minimum pointu devrait avoir une sharpness élevée."""
    from src.metrics import compute_sharpness

    base_loss = 0.5
    perturbed = [1.5, 0.9, 0.5, 0.9, 1.5]
    sharpness = compute_sharpness(base_loss, perturbed)

    assert sharpness > 0.3, f"Sharpness trop faible pour un min pointu : {sharpness}"
