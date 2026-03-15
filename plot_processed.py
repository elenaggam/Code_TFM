import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
dir = base + "Triangle/0/interpolation_naive_BR/5_35/sklearn_nnmf/8_components.hspy"
# , base + "Triangulos/", base + "Estrellas/"

s = hs.load(dir)
s.plot()
plt.show()