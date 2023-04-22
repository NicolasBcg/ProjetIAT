import torch
from torch import nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, ny: int, nx: int, nf: int, na: int):
        """À MODIFIER QUAND NÉCESSAIRE.
        Ce constructeur crée une instance de réseau de neurones convolutif (CNN).
        L'architecture choisie doit être choisie de façon à capter toute la complexité du problème
        sans pour autant devenir intraitable (trop de paramètres d'apprentissages). 

        :param na: Le nombre d'actions 
        :type na: int
        """
        super(CNN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(nf, 32, 3, stride=1, padding="same", bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding="same", bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(ny * nx * 32, 120),
            nn.Linear(120, na)
        )

        self.layers.apply(weights_init)

    def forward(self, x):
        """Cette fonction réalise un passage dans le réseau de neurones. 

        :param x: L'état
        :return: Le vecteur de valeurs d'actions (une valeur par action)
        """
        qvalues = self.layers(x)
        return qvalues

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)