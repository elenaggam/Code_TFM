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
base = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_grilla_SiN/"
dir_zlp = base + 'ZLP.dm4'
list_directories =  [base + "Estrellas/"]
# , base + "Triangulos/"
suffix = '8_33_sklearn_pca/'

# noise removal parameters + mask
pre_treatment = True

background = {
    "done": False,
    "name_bg": 'interpolation',
    "name_ic": 'naive'
}

components_method = 'NMF'
reduce = True

E = {
    "low": [0.8],
    "high": [3.3]
}

mask = {
    "done": False,
    "method": 'otsu',
    "merge": True
}

#n_components = np.loadtxt(list_directories[0]+'pca_835.txt', dtype=int)




def full_data_treatment(list_directories, background, mask, components_method, E, suffix=None, n_components=None, save_components=False, plot_components=True, bss=False, bss_components=[None, None, None, None, None, None, None, None, None, None,None, None, None, None, None]):
    '''
    Apply the full noise filtering process to the data in the specified directories, 
    including background removal, mask generation and dimensionality reduction.
    
    Parameters:
    -----------
    list_directories : list of str
        List of directories containing the data to be processed. Each directory should contain .dm4 files.
    background : dict
        Dictionary containing the parameters for background removal.
    mask : dict
        Dictionary containing the parameters for mask generation.
    components_method : str
        The method to use for dimensionality reduction.
    E : dict
        Dictionary containing the energy range for dimensionality reduction.
    suffix : str, optional
        Suffix for the output directory name.
    n_components : int, optional
        Number of components for NMF.
    save_components : bool, optional
        Whether to save the components.
    plot_components : bool, optional
        Whether to plot the components.

    Returns:
    --------    
    None
    '''
    
    for data_path_base in list_directories: 
        # align zlp and get a background spectrum for eacd dm4 file
        if not pre_treatment:
            df.dm4_pre_treatment(data_path_base)
        dm4_files = [f for f in os.listdir(data_path_base) if f.endswith('.dm4')]
        aligned_files = [f for f in os.listdir(data_path_base) if f.endswith('Aligned.hspy')]
        back_files = [f for f in os.listdir(data_path_base) if f.endswith('b.hspy')]
        
        for m in range(len(dm4_files)): 
            
            dm4_file = dm4_files[m]

            # folder for each file, according to the bacjground removal method
            if background["name_bg"] == 'thesis':
                dir = data_path_base + f'{m+1}_{background["name_bg"]}_BR/'
            else:
                dir = data_path_base + f'{m+1}_{background["name_ic"]}_BR/'
            print(dir)
            if not os.path.exists(dir):
                os.makedirs(dir)

            # binary mask
            if not mask["done"]:
                data_scan = hs.load(os.path.join(data_path_base, dm4_file))[1]
                dir_mask = data_path_base + f'{m+1}-mask/' #masks folder, as they dont depend on the background method
                if not os.path.exists(dir_mask):
                    os.makedirs(dir_mask)
                if mask["merge"]:
                    df.get_mask(data_scan, dir_mask, method='otsu')
                    df.get_mask(data_scan, dir_mask, method='adaptive')
                    df.merge_masks(dir_mask)
                else:
                    df.get_mask(data_scan, dir_mask, method=mask["method"])
                del data_scan

            # background removal
            if not background["done"]:
                # load the original spectrum and background
                s = hs.load(os.path.join(data_path_base, aligned_files[m]))
                b = hs.load(os.path.join(data_path_base, back_files[m]))
                if background["name_bg"] == 'thesis':
                    s2 = df.background_thesis(s, b)
                else:
                    aux = s.deepcopy()
                    s2 = df.intensity_correction(aux-b, name=background["name_ic"])
                    del aux
                s2.save(dir + 'Data.hspy', overwrite=True)
                del s, b, s2

            # first folder, named after the background and interpolation method
            # suffix is used in case we want to do a second iteration (new or over the existing denoised data) of the process with different parameters
            if suffix is None:
                dir_output = dir 
            else:
                dir_output  = dir + suffix

            # dimensionality reduction
            # load the background removed spectrum
            for file in os.listdir(dir_output): 
                if file.endswith('Data.hspy'): 
                    new = hs.load(os.path.join(dir_output, file))
                    break # just one file should be found, but in case there are more, we take the first one
            
            # select the energy range for the dimensionality reduction, try different ranges
            if reduce:
                for E_low in E["low"]:
                    for E_high in E["high"]:
    
                        str_E = f'{int(E_low*10)}_{int(E_high*10)}'
                        new = new.isig[E_low:E_high]
                        if suffix:
                            dir_output_E = dir_output + components_method + '/'
                        else:
                            dir_output_E = dir_output + str_E + '_' + components_method + '/'
                        
                        if bss:
                            dir_output_E += 'BSS/'
                        
                        # if n_components is not None, add it to the folder name, since we iterate over it in NMF
                        if n_components is not None and components_method == 'NMF':
                            dir_output_E += f'{n_components}/'
                        
                        if not os.path.exists(dir_output_E+'logs/'):
                            os.makedirs(dir_output_E+'logs/') 
    
                        # apply dimensionality reduction to the background removed spectrum
                        t_init = time.time()
                        print(f'Initializing component analysis for {dir_output_E}')
    
                        df.components_reduction(dir_output_E, new, method=components_method, n_components=n_components, save_components=save_components, plot_components=plot_components, bss=bss, bss_components=bss_components[m])
    
                        t_end = time.time()
                        file = open(dir_output_E + 'logs/time.txt', 'w') 
                        file.write(f"{t_end - t_init:.2f}") 
                        file.close()
            
    print('DONE')

    return

full_data_treatment(list_directories, background, mask, 'sklearn_pca', E, n_components = 10)
#s = hs.load(list_directories[0]+'1_interpolation_naive_BR/8_35_sklearn_pca/5_components_Data.hspy')

#df.components_reduction(list_directories[0]+'1_interpolation_naive_BR/8_35_sklearn_pca/NMF/BSS/', s, method='NMF',n_components=3, bss=True, bss_components=5)
        
    