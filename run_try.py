import functions as F
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
data_path_base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
data_path = data_path_base + 'Triangle/4-STEM SI.dm4'
s = hs.load(data_path)[-1]
load_t = time.time()
print(f"\nData loaded in {(load_t - start):.0f} s\n")
del load_t
del start

E = (3., s.axes_manager[2].axis[-1])
avg, x, y=F.best_avg_roi(s, nx=10, ny=10, threshold=0, E=E, x0=s.axes_manager[0].axis[-1]/2, steps_x=5, steps_y=5)
print(f"\nBest average found: {avg:.2f} counts in ROI x:{x}, y:{y}\n")

new=F.background_removal(s, x, y)
new=F.intensity_correction(new, threshold=0, name='naive')
new.plot()
plt.show()


# new.decomposition("sklearn_pca") 
# #new.decomposition(True) 
# new.learning_results.summary()
# new.plot_explained_variance_ratio(n=40)
# a = new.estimate_elbow_position(explained_variance_ratio=None, log=True, max_points=40)
# print(f'Componentes principales: {a}')
# new.plot_decomposition_results()
# new.blind_source_separation(4, diff_order=1)
# _ = new.plot_bss_loadings()
# _ = new.plot_bss_factors()
# sm=new.get_decomposition_model(2*a)
# sm.decomposition(True, algorithm="NMF", output_dimension=2*a,max_iter=50000)
# #sm.plot_decomposition_results()
# sm.learning_results.summary()
# _ = sm.plot_decomposition_loadings()
# _ = sm.plot_decomposition_factors()
# sm.T.plot()
# A = sm.get_decomposition_loadings()

# B =sm.get_decomposition_factors().as_signal1D(1)