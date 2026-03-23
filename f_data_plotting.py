import matplotlib.pyplot as plt
import hyperspy.api as hs
import os
import numpy as np
from scipy.ndimage import gaussian_filter 

def plot_component(dir):

    comp = hs.load(dir)
    copy = comp.copy()
    copy.add_gaussian_noise(std=1e-16)
    copy.plot(navigator_kwds={'cmap': 'turbo', 'vmax': 1.3e-11, 'vmin':0.2e-11})
    plt.show()

base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
dir = base + "Triangulos/2/"
# , base + "Triangulos/", base + "Estrellas/"
plot_component(dir+'3.hspy')
#plot_component(dir+'8.hspy')