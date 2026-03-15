import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as df
import sys

data_path_base  = str(sys.argv[1])
dm4_file = str(sys.argv[2])
file_index = int(sys.argv[3])
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
print(f"Total time taken: {t_end - t_init:.2f} seconds")