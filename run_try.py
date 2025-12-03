import functions as F
import hyperspy.api as hs
import matplotlib.pyplot as plt

data_path_base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
data_path = data_path_base + 'Triangle/4-STEM SI.dm4'
s = hs.load(data_path)[-1]

E = (3., s.axes_manager[2].axis[-1])
avg, x, y=F.best_avg_roi(s, nx=10, ny=10, threshold=0, E=E, x0=s.axes_manager[0].axis[-1]/2, steps_x=10, steps_y=10)
print(f"\nBest average found: {avg} counts in ROI x:{x}, y:{y}\n")

new=F.background_removal(s, x, y, E=E)
new.plot()
plt.show()

