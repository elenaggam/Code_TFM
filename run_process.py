import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as df


base = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_Grilla_SiN/"
list_directories = [base + "Triangle/", base + "Triangulos/", base + "Estrellas/"]
# 

background = {
    "name_bg": 'interpolation',
    "name_ic": 'naive'
}

components_method = 'sklearn_pca'

for data_path_base in list_directories: 
    dm4_files = [f for f in os.listdir(data_path_base) if f.endswith('.dm4')]

    for dm4_file in dm4_files: 
        file_index = dm4_files.index(dm4_file)
        dir_output = data_path_base + f'{file_index}/' + background["name_bg"] + '_' + background["name_ic"] + '_BR/'

        # load the original spectrum
        original = hs.load(os.path.join(data_path_base, dm4_file))[-1]
        print(f'Loaded file: {dm4_file}')

        # remove the background from the original spectrum and save the result
        df.background_removal(dir_output, original)

        # load the background removed spectrum
        for file in os.listdir(dir_output): 
            if file.endswith('Data.hspy'): 
                new = hs.load(os.path.join(dir_output, file))
        print(f'Background removed spectrum loaded: {file}')

        # apply dimensionality reduction to the background removed spectrum
        t_init = time.time()

        df.components_reduction(dir_output, new, method=components_method)

        t_end = time.time()
        file = open(dir_output + 'logs/' + components_method + '_time.txt', 'w') 
        file.write(f"{t_end - t_init:.2f}") 
        file.close()
        
        for f in os.listdir(dir_output): 
            if f.endswith('components.hspy'): 
                new = hs.load(os.path.join(dir_output, f))

        df.otsu_mask(new, dir_output + components_method + '_')

# dir_output = base + "Triangle/" + f'{0}/' + background["name_bg"] + '_' + background["name_ic"] + '_BR/'
# for file in os.listdir(dir_output): 
#     if file.endswith('data.hspy'): 
#         new = hs.load(os.path.join(dir_output, file))
# t_init = time.time()
# df.components_reduction(dir_output, new, method=components_method)
# t_end = time.time()
# file = open(dir_output + 'logs/' + components_method + '_time.txt', 'w') 
# file.write(f"{t_end - t_init:.2f}") 
# file.close()