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
│   ├── visualization.py    # Heatmap P02 + importance des hyperparamètres
│   ├── loss_landscape_analysis.py # Analyse approfondie du loss landscape 1D
│   └── metrics.py          # F1, gap train/val, sharpness
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_analysis.ipynb
├── tests/
│   └── test_data_loader.py
├── results/
│   ├── optuna_study.db     # Base SQLite de l'étude Optuna
│   ├── optuna_study.pkl    # Étude Optuna sérialisée
│   ├── optuna_trials.csv   # Historique détaillé des trials
│   ├── best_params.json    # Meilleure config trouvée par Optuna
│   ├── grid_p02_results.csv# Résultats grille exhaustive (12 combinaisons)
│   ├── checkpoints/        # Meilleurs modèles sauvegardés
│   └── figures/            # Toutes les figures générées (heatmap, importance, loss landscape, ...)
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

### 1. Optimisation P02 (Optuna +/− grille exhaustive)

Depuis la racine du projet, avec l'environnement Poetry activé :

```bash
# Étude Optuna seule (recherche Bayésienne)
poetry run run-optimization --mode optuna --n-trials 20 --output results

# Grille déterministe seule (12 combinaisons weight_decay × dropout)
poetry run run-optimization --mode grid --output results

# Optuna + grille (pour avoir à la fois best_params et grid_p02_results.csv)
poetry run run-optimization --mode both --n-trials 20 --output results
```

Les artefacts principaux sont écrits dans `results/` :
- `optuna_study.db`, `optuna_study.pkl`, `optuna_trials.csv`, `best_params.json`
- `grid_p02_results.csv` pour la grille complète P02

### 2. Visualisations Optuna (heatmap + importance des hyperparamètres)

```bash
# Importance des hyperparamètres (Optuna)
poetry run python -m src.visualization --study results/optuna_study.pkl

# Heatmap gap / F1 + importance (si la grille a été calculée)
poetry run python -m src.visualization \
	--study results/optuna_study.pkl \
	--grid-csv results/grid_p02_results.csv
```

Les figures sont sauvegardées dans `results/figures/` (par ex. `optuna_importance.png`, `heatmap_p02.png`).

### 3. Analyse du loss landscape 1D (minima plats vs pointus)

Pour comparer la géométrie du minimum entre plusieurs configurations de régularisation :

```bash
poetry run run-landscape-loss
```

Ce script :
- entraîne rapidement plusieurs configurations (sous‑régularisée, baseline, très régularisée, + meilleure config Optuna si disponible) sur un sous‑ensemble d'Allociné ;
- calcule un loss landscape 1D autour du minimum pour chaque configuration ;
- génère une figure de comparaison dans `results/figures/loss_landscape_1d.png` avec la *sharpness* de chaque minimum.

### 4. Dashboard Optuna 

Si `pip install optuna-dashboard` est installé dans ton environnement :

```bash
optuna-dashboard sqlite:///results/optuna_study.db
```

Cela ouvre une interface web interactive pour explorer tous les trials Optuna.

### 5. Lancer les notebooks
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

**Auteurs du projet :**
- NGOULOU NGOUBILI Irch
- MOYO Guillaine
- DOMEVENOU Wisdom

**Contact encadrant :**
- mbialaura12@gmail.com

Soumission : Rapport PDF + lien GitHub par mail  
**Date limite : 13 mars 2026**
