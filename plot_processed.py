import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import f_data_filtering as df

base = "C:/Users/PC_TEM/Documents/Elena/Au_NPs_Grilla_SiN/"




suffix = "interpolation_naive_BR/8_33/sklearn_pca/"
list_dir = [base + "Triangle/0/"+suffix, 
            base + "Triangulos/0/"+suffix,
            base + "Triangulos/1/"+suffix,
            base + "Triangulos/2/"+suffix,
            base + "Triangulos/3/"+suffix]


# , base + "Triangulos/", base + "Estrellas/"
comp_file = base + 't_pca_components_833.txt'
  
dir = list_dir[0]
def plot_component(dir, E=None, noise = 1e-14):

    comp = hs.load(dir)
    if E:
        copy = comp.copy().isig[E[0]:E[1]]
    else:
        copy = comp.copy()
    copy.add_gaussian_noise(std=noise)
    copy2 = copy.rebin(scale=(2,2,1))
    #copy2.data[copy2.data<=0]=1e-10 
    #'vmin':1.0e2, 'vmax': 1e4,'norm':"log"
    copy2.plot(navigator_kwds={'cmap': 'jet', 'vmin':1.0e-12})
    # , 'vmax': 1.2e-11, 'vmin':0.0e-11
    plt.show()


plot_component(dir+'components/3.hspy', (1.1,1.7))
plot_component(dir+'components/12.hspy', (1.7, 2.3))
    
'''
s = hs.load(list_dir[0]+'rebuilt_all.hspy')
s.plot()
plt.show()

sx =hs.load(base+'Triangle/0/interpolation_naive_BR/Data.hspy').isig[0.8: 3.3]
sx.plot()
plt.show()

for i in range(len(list_dir)):
    with open(comp_file) as f:
        for line in f:
            tips = []
            sides = []
            # read each row of the components file
            row = [int(x) for x in line.split()]
            del row[:2]
            j = row[0]
            for l in row:
                if l != j:
                    j = l
                    sides.append(j)
                else:
                    tips.append(l)
                j +=1
            df.components_rebuild(list_dir[i], tips, 'tips')
            df.components_rebuild(list_dir[i], sides, 'sides')
            t = hs.load(list_dir[i]+'tips.hspy')
            t.plot()
            plt.show()
            s = hs.load(list_dir[i]+'sides.hspy')
            s.plot()
            plt.show()
                
'''
# managua, berlin, vanimo
#tip = df.components_rebuild(dir, sides).rebin(scale=[2,2,1])
#tip.plot(navigator_kwds={'cmap': 'Greens', 'vmax': 1.0e-20, 'vmin':-1.0e-20})
#plt.show()

