import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as df


base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
list_directories = [base + "Triangulos/", base + "Estrellas/"]
# base + "Triangle/",

for data_path_base in list_directories: 
    dm4_files = [f for f in os.listdir(data_path_base) if f.endswith('.dm4')]

    for dm4_file in dm4_files: 
        file_index = dm4_files.index(dm4_file)
        dir_output = data_path_base + f'{file_index}/'

        # load the original spectrum
        original = hs.load(os.path.join(data_path_base, dm4_file))[-1]
        print(f'Loaded file: {dm4_file}')

        # store the background removed spectrum in a new directory
        if not os.path.exists(dir_output): 
            os.makedirs(dir_output)
        df.background_removal(dir_output, original)

        # load the background removed spectrum
        for file in os.listdir(dir_output): 
            if file.endswith('_background_removed.hspy'): 
                new = hs.load(os.path.join(dir_output, file))
        print(f'Background removed spectrum loaded: {file}')

        # apply dimensionality reduction to the background removed spectrum
        t_init = time.time()

        df.components_reduction(dir_output, new)

        t_end = time.time()
        file = open(dir_output + 'PCA_time.txt', 'w') 
        file.write(f"\nPCA completed in {t_end - t_init:.2f} seconds ({(t_end - t_init)/60:.2f} minutes).\n") 
        file.close()

dir_output = base + "Triangle/"
for file in os.listdir(dir_output): 
    if file.endswith('_background_removed.hspy'): 
        new = hs.load(os.path.join(dir_output, file))
t_init = time.time()
df.components_reduction(dir_output, new)
t_end = time.time()
file = open(dir_output + f'PCA_time.txt', 'w') 
file.write(f"\nPCA completed in {t_end - t_init:.2f} seconds ({(t_end - t_init)/60:.2f} minutes).\n") 
file.close()