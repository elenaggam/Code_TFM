import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import f_data_filtering as df

base = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_Grilla_SiN/"

dir_zlp = base + 'ZLP.dm4'

suffix = "interpolation_naive_BR/8_33/sklearn_pca/rebuilt_all.hspy"
list_dir = [base + "Triangle/0/"]


E = [.8, 3.3]
zlp = hs.load(dir_zlp)[0].isig[E[0]:E[1]]


for i in range(len(list_dir)):
    dir_mask = (list_dir[i] + 'otsu_mask.txt')
    s = hs.load(list_dir[i] + suffix).rebin(new_shape=[422,321,167])
    zlp.axes_manager[-1].scale = s.axes_manager[-1].scale
    zlp.axes_manager[-1].offset = s.axes_manager[-1].offset
    zlp.axes_manager[-1].units = s.axes_manager[-1].units
    df.dielectric_function(s, zlp.data.sum(), list_dir[i], dir_mask)