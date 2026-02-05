import functions as F
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import time

data_path_base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
# data_path_basis = [data_path_base + 'Triangulos/2-STEM SI', data_path_base + 'Triangulos/3-STEM SI', data_path_base + 'Triangulos/4-STEM SI', data_path_base + 'Triangulos/STEM SI', data_path_base + 'Triangle/2-STEM SI aligned.hspy', data_path_base + 'Estrellas/2-STEM SI', data_path_base + 'Estrellas/3-STEM SI', data_path_base + 'Estrellas/STEM SI']

# for path in data_path_basis:
#     print(f'Processing file: {path}')
#     s = hs.load(path+'.dm4')[-1]
#     s.align_zero_loss_peak()
#     s.save(path+' aligned.hspy')

#     del s


start = time.time()
data_path = data_path_base + 'Triangle/4-STEM SI aligned.hspy'
s = hs.load(data_path)
# s = F.normalize_pixelwise(s)
load_t = time.time()
print(f"\n\nData loaded in {(load_t - start):.0f} s")

x = (185.67435350915267, 195.67435350915267)
y = (99.48078605699243, 109.48078605699243)

a = s.axes_manager[0].axis[-1]
b = s.axes_manager[1].axis[-1]
x2 = (a-10, a)
y2 = (b-10, b)


# s.plot()
# plt.show()

# new = F.rebin(s.isig[:4.], ne=3)
# roi = hs.roi.RectangularROI(left=x[0], right=x[1], top=y[0], bottom=y[1])
# b = np.mean(roi(new).data, axis=(0,1))s

b = F.background_constant(s, x, y)
plt.plot(s.axes_manager[2].axis, b)
S = F.background_substraction_thesis(s, b)
S.plot()
plt.show()
S.rebin(scale=[6,6,2]).isig[0.8:4.].plot()
plt.show()
# new = s-b
# new.rebin(scale=[6,6,2]).isig[0.8:4.].plot()
# plt.show()   
# roi = hs.roi.RectangularROI(left=99., right=109., top=38., bottom=48.)
# roi(new).plot()
# plt.show()




# start = load_t

# s2 = F.normalize_pixelwise(s)
# load_t = time.time()
# print(f"\n\nData normalized in {(load_t - start):.0f} s")
# del load_t
# del start
# s2.plot()
# plt.show()



# new = F.background_removal(s2, x, y)
# print(new.data.min(), new.data.max())
# new.plot()
# plt.show()



# file = open('trial best avg2.txt', 'w')
# E = (3., s.axes_manager[2].axis[-1])
# steps_x = int((s.axes_manager[0].axis[-1]-s.axes_manager[0].axis[0]))
# steps_y = int((s.axes_manager[1].axis[-1]-s.axes_manager[1].axis[0]))
# avg, x, y=F.best_avg_roi(s, nx=10, ny=10, E=E, steps_x=steps_x, steps_y=steps_y)
# print(f"\nBest average found: {avg:.2f} counts in ROI x:{x}, y:{y}\n")
# file.write(f'{x[0]}\t{x[1]}\n{y[0]}\t{y[1]}\n{avg}')
# file.close()





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

