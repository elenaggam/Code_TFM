import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as df
import datetime
import sys

print(f'\n\nLanzado en {datetime.datetime.now()}. Revisa out.log para la salida.\n')

# directories 
base_a = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_grilla_SiN/"
base_b = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_grilla_Carbono/"

list_directories =  [ base_a + "Triangulos/", base_a + "Estrellas/",base_b + "Estrellas/", base_b + "Triangulos/"]

background = {
    "done": False,
    "name_bg": 'thesis',
    "name_ic": 'naive'
}

dim_reduction = {
    'method': 'sklearn_pca',
    'n_components': 16,
    'folder_name': '8_33_sklearn_pca/',
    'plot_components': True,
    'save_components': False
}


mask = {
    "done": False,
    "method": 'otsu',
    "merge": True
}

#n_components = np.loadtxt(list_directories[0]+'pca_835.txt', dtype=int)

for directory in list_directories:
    print(f'\n\nProcesando {directory}...\n')
    df.full_data_treatment(directory, background, mask, dim_reduction, E_slice=(0.8, 3.3))

