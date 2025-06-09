import numpy as np

# Calcul de la précision
def precision(y,y_gt):
    return (y.argmax(0)==y_gt.argmax(0)).mean()

# Fonction de coût : calcule l'erreur quadratique moyenne
def loss(y,y_gt): # y est obtenue en sortie du perceptron et y_gt est la sortie attendue
    return (0.5 * (y-y_gt)**2).sum(0).mean()

def loss_back(y,y_gt):
    return (y-y_gt)/y.shape[1] # dérivée de la fonction de coût

# Conversion de la classe en représentation one hot
def one_hot(y, num_classes=10):
  # Crée une matrice de zéros de taille (nombre d'échantillons, nombre de classes)
    z = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    z[np.arange(y.shape[0]), y] = 1.
    return z

def to_tensor(x):
  # Normalise les valeurs de pixel entre 0 et 1 et redimensionne l'image en un vecteur
    return x.reshape((x.shape[0], -1)).astype(np.float32) / 255.
