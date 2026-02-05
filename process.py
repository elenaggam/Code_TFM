import functions as F
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import data_filtering as df

data_path_base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/Triangle/"

original = hs.load(data_path_base+'4-STEM SI.dm4')[-1]
#df.background_removal(data_path_base+'_', original)

for file in os.listdir(data_path_base):
    if file.endswith('_background_removed.hspy'):
        print(f'Loading processed file: {file}')
        new = hs.load(os.path.join(data_path_base, file))
        break
original.save(data_path_base+'4-STEM SI aligned.msa')
original.plot()
plt.show()
new.plot()
plt.show()