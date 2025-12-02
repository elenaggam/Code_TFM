import functions as F
import hyperspy.api as hs
import matplotlib.pyplot as plt

data_path_base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
data_path = data_path_base + 'Triangle/4-STEM SI.dm4'
s = hs.load(data_path)[-1]

new=F.background_removal(s, (200, 210), (150, 160), E=(-2.,3.))


new3=F.ic_averaged(new, threshold=500,  E=(3., new.axes_manager[2].axis[-1]))
new3.plot()
plt.show(block=False)



input("Press Enter to continue...")