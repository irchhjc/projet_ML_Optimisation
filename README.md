# G10 — CamemBERT × Allociné : Régularisation & Généralisation

## 🎯 Problématique (P02)

> **Comment le weight decay et le dropout affectent-ils la généralisation de CamemBERT sur les critiques de films en français ?**

| Axe | Valeur |
|-----|--------|
| Dataset | D05 — Allociné (100k critiques FR, 2 classes) |
| Modèle | M04 — CamemBERT-base (110M paramètres) |
| Problématique | P02 — Régularisation et Généralisation |
| Méthode d'optimisation | Optuna (Bayésien) |
| Métrique principale | F1-score (macro) |

---

## 📂 Structure du projet

```
g10_camembert/
├── pyproject.toml          # Dépendances Poetry
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparamètres et chemins
│   ├── data_loader.py      # Chargement et sous-échantillonnage Allociné
│   ├── model_setup.py      # Init CamemBERT avec dropout configurable
│   ├── trainer.py          # Boucle d'entraînement custom
│   ├── optimization.py     # Étude Optuna (weight decay × dropout)
│   ├── visualization.py    # Loss landscape + courbes de généralisation
│   └── metrics.py          # F1, gap train/val, sharpness
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_analysis.ipynb
├── tests/
│   └── test_data_loader.py
├── results/
│   └── figures/
└── report/
```

---

## 🚀 Installation

```bash
# Installer Poetry (si nécessaire)
curl -sSL https://install.python-poetry.org | python3 -

# Installer les dépendances
poetry install

# Activer l'environnement
poetry shell
```

---

## ▶️ Exécution

### 1. Optimisation Optuna (recherche de weight_decay × dropout)
```bash
poetry run python -m src.optimization --n-trials 20 --output results/optuna_study.pkl
```

### 2. Analyse du Loss Landscape
```bash
poetry run python -m src.visualization --checkpoint results/best_model --output results/figures/
```

### 3. Entraînement final avec les meilleurs hyperparamètres
```bash
poetry run python -m src.trainer --config results/best_params.json --mode final
```

### 4. Lancer les notebooks
```bash
poetry run jupyter notebook notebooks/
```

---

## 🧪 Tests
```bash
poetry run pytest tests/ -v --cov=src
```

---

## ⚙️ Adaptation CPU

Le projet est entièrement conçu pour fonctionner **sans GPU** :
- Sous-échantillonnage équilibré (500 train / 150 val par défaut)
- `gradient_accumulation_steps` pour simuler des grands batch sizes
- Quantification dynamique disponible (`src/model_setup.py`)
- Loss landscape 1D léger (8 points, ε = 0.05)
- Early stopping pour limiter le temps de calcul

---

## 📊 Résultats attendus

Le protocole étudie un **grid search** via Optuna sur :
- `weight_decay` ∈ {1e-5, 1e-4, 1e-3, 1e-2}
- `dropout` ∈ {0.0, 0.1, 0.3}
- → 12 combinaisons + exploration Bayésienne

**Métriques rapportées :**
- F1-score macro (test)
- Gap généralisation = F1_train − F1_val
- Sharpness du minimum (analyse loss landscape)

---

## 📬 Contact

- Enseignant : mbialaura12@gmail.com
- Soumission : Rapport PDF + lien GitHub par mail
- **Date limite : 13 mars 2026**
