import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as df


base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
list_directories = [base + "Triangle/"]
# , base + "Triangulos/", base + "Estrellas/"

background = {
    "name_bg": 'interpolation',
    "name_ic": 'naive'
}

E = [.8, 4.5]
str_E = '8_45'

dir_zlp = base + 'ZLP.dm4'

components_method = 'sklearn_nnmf'
mask_method = 'adaptive'

back_done = True
mask_done = True
merge_mask = False

for data_path_base in list_directories: 
    dm4_files = [f for f in os.listdir(data_path_base) if f.endswith('.dm4')]

    for dm4_file in dm4_files: 
        file_index = dm4_files.index(dm4_file)
        dir = data_path_base + f'{file_index}/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir_output = dir + background["name_bg"] + '_' + background["name_ic"] + '_BR/'
        
        # binary mask
        if not mask_done:
            data_scan = hs.load(os.path.join(data_path_base, dm4_file))[1]
            df.get_mask(data_scan, dir, method=mask_method)
            if merge_mask:
                df.merge_masks(dir)
            del data_scan
        
        # background removal
        if not back_done:
            # load the original spectrum
            original = hs.load(os.path.join(data_path_base, dm4_file))[-1]
            print(f'Loaded file: {dm4_file}')
    
            # remove the background from the original spectrum and save the result
            df.background_removal(dir_output, original)
            del original

        # dimensionality reduction
        # load the background removed spectrum
        for file in os.listdir(dir_output): 
            if file.endswith('Data.hspy'): 
                new = hs.load(os.path.join(dir_output, file))
        print(f'Background removed spectrum loaded: {file}')
        
        if E: 
            new = new.isig[E[0]:E[1]]
            dir_output = dir_output + str_E + '/'
            if not os.path.exists(dir_output):
                os.makedirs(dir_output) 
            if not os.path.exists(dir_output+'logs/'):
                os.makedirs(dir_output+'logs/')
                
        # apply dimensionality reduction to the background removed spectrum
        t_init = time.time()

        dir_output += components_method + '/'
        df.components_reduction(dir_output, new, method=components_method, n_components=8)
        del new

        t_end = time.time()
        file = open(dir_output + 'logs/' + components_method + '_time.txt', 'w') 
        file.write(f"{t_end - t_init:.2f}") 
        file.close()
        
        for f in os.listdir(dir_output): 
            if f.endswith('components.hspy'): 
                s = hs.load(os.path.join(dir_output, f))



