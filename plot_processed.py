import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

dir = 'c'

s = hs.load(dir)
s.plot()
plt.show()