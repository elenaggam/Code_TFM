import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as f
import datetime
import sys

print(f'\n\nLanzado en {datetime.datetime.now()}. Revisa out.log para la salida.\n')

# directories 
base_a = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_grilla_SiN/"
base_b = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_grilla_Carbono/"

list_directories =  [ base_a + "Triangulos/", base_a + "Estrellas/",base_b + "Estrellas/", base_b + "Triangulos/"]

background = {
    "done": True,
    "name_bg": 'thesis',
    "name_ic": 'naive'
}

dim_reduction = {
    'method': 'NMF',
    'n_components': 16,
    'folder_name': '8_33_sklearn_pca_nmf/',
    'plot_components': True,
    'save_components': True
}


mask = {
    "done": True,
    "method": 'otsu',
    "merge": True
}

#n_components = np.loadtxt(list_directories[0]+'pca_835.txt', dtype=int)

for directory in list_directories:

    dm4_files = [f for f in os.listdir(directory) if f.endswith('.dm4')]
    
    for i in range(len(dm4_files)):
        base = directory + f'{i+1}_thesis_naive_BR/'
        data_path_base = directory + f'{i+1}_thesis_naive_BR/8_33_sklearn_pca/'
        
        for file in os.listdir(data_path_base):
            if file.endswith('Data.hspy'):
                dm4 = file
                break
        print(f'Processing {dm4}...')
        s = hs.load(os.path.join(data_path_base, dm4))
        
        f.components_reduction(base + dim_reduction['folder_name'], s, method=dim_reduction['method'], n_components=dim_reduction['n_components'],  plot_components=dim_reduction['plot_components'], save_components=dim_reduction['save_components'])

        print(f'Finished processing {dm4} at {datetime.datetime.now()}.\n')
    print(f'Finished processing all files in {data_path_base} at {datetime.datetime.now()}.\n')