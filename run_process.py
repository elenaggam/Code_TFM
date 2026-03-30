import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as df
import datetime

print(f'\n\n\nLanzado en {datetime.datetime.now()}. Revisa out.log para la salida.\n')

# directories 
base = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_grilla_Carbono/"
dir_zlp = base + 'ZLP.dm4'
list_directories =  [ base + "Triangulos/", base + "Estrellas/"]
# [base + "Triangle/",
suffix = '8_32/sklearn_pca/'

# noise removal parameters + mask
background = {
    "done": False,
    "name_bg": 'interpolation',
    "name_ic": 'naive'
}

components_method = 'sklearn_pca'

E = {
    "low": [0.5, 0.8],
    "high": [3.5]
}

mask = {
    "done": False,
    "method": 'otsu',
    "merge": True
}

# trying different numbers of components in NMF
list_c = [i for i in range(2, 21)]


def full_data_treatment(list_directories, background, mask, components_method, E, dm4_files=None, suffix=None, n_components=None, save_components=False, plot_components=True):
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
    dm4_files : list of str, optional
        List of .dm4 files to process. If None, all .dm4 files in the directories will be processed.
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
        if dm4_files is None:
            dm4_files = [f for f in os.listdir(data_path_base) if f.endswith('.dm4')]

        for dm4_file in dm4_files: 
            file_index = dm4_files.index(dm4_file)
            # folder for each file
            dir = data_path_base + f'{file_index}/'
            if not os.path.exists(dir):
                os.makedirs(dir)

            # binary mask
            if not mask["done"]:
                data_scan = hs.load(os.path.join(data_path_base, dm4_file))[1]
                if mask["merge"]:
                    df.get_mask(data_scan, dir, method='otsu')
                    df.get_mask(data_scan, dir, method='adaptive')
                    df.merge_masks(dir)
                else:
                    df.get_mask(data_scan, dir, method=mask["method"])
                del data_scan

            # first folder, named after the background and interpolation method
            # suffix is used in case we want to do a second iteration (new or over the existing denoised data) of the process with different parameters
            if suffix is None:
                dir_output = dir + background["name_bg"] + '_' + background["name_ic"] + '_BR/'
            else:
                dir_output  = dir + background["name_bg"] + '_' + background["name_ic"] + '_BR_max/' + suffix
            
            # background removal
            if not background["done"]:
                # load the original spectrum
                original = hs.load(os.path.join(data_path_base, dm4_file))[-1]
                print(f'Loaded file: {dm4_file}')
                # remove the background from the original spectrum and save the result
                df.background_removal(dir_output, original, E=(0.,1.))
                del original

            # dimensionality reduction
            # load the background removed spectrum
            for file in os.listdir(dir_output): 
                if file.endswith('.hspy'): 
                    print(file)
                    new = hs.load(os.path.join(dir_output, file))
                    break # just one file should be found, but in case there are more, we take the first one
            print(f'Background removed spectrum loaded: {file}') 
            
            # select the energy range for the dimensionality reduction, try different ranges
            for E_low in E["low"]:
                for E_high in E["high"]:

                    str_E = f'{int(E_low*10)}_{int(E_high*10)}'
                    new = new.isig[E_low:E_high]
                    dir_output_E = dir_output + str_E + '/' + components_method + '/'
                    
                    # if n_components is not None, add it to the folder name, since we iterate over it in NMF
                    if n_components is not None and components_method == 'NMF':
                        dir_output_E += f'{n_components}/'
                    
                    if not os.path.exists(dir_output_E):
                        os.makedirs(dir_output_E) 
                    if not os.path.exists(dir_output_E+'logs/'):
                        os.makedirs(dir_output_E+'logs/')

                    # apply dimensionality reduction to the background removed spectrum
                    t_init = time.time()
                    print(f'Initializing component analysis for {dir_output_E}')

                    df.components_reduction(dir_output_E, new, method=components_method, n_components=n_components, save_components=save_components, plot_components=plot_components)

                    t_end = time.time()
                    file = open(dir_output_E + 'logs/time.txt', 'w') 
                    file.write(f"{t_end - t_init:.2f}") 
                    file.close()

    print('DONE')

    return


full_data_treatment(list_directories, background, mask, components_method, E)

