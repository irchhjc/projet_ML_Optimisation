#%%
import matplotlib.pyplot as plt

import torch
from torch import nn  # sous module de torch qui contient les fonctions de perte, les couches, etc.
from torch.nn import functional as F  # contient les fonctions d'activation, de perte, etc.

from torchvision import datasets, transforms  # pour les jeux de données et les transformations d'images
#%%

X = torch.tensor(3., requires_grad=True)  # crée un tenseur avec la valeur 3 et indique que nous voulons calculer les gradients
Y  = torch.tensor(3., requires_grad=True)  # crée un autre tenseur avec la valeur 3 et indique que nous voulons calculer les gradients

def f(x, y):
    return x**2 + y**2  # définit une fonction simple qui prend deux entrées et retourne la somme de leurs carrés

# Backpropagation
res = f(X, Y)  # calcule la valeur de la fonction f pour les tenseurs X et Y
res.backward()  # calcule les gradients de res par rapport à X et Y

X -= 0.1 * X.grad  # met à jour X en soustrayant un petit pas de la valeur du gradient de X
Y -= 0.1 * Y.grad  # met à jour Y en soustrayant un petit pas de la valeur du gradient de Y
#%%











# %%
