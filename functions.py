import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs


# Background substraction for EELS spectra

def background_constant(s, x, y):
    """
    Subtracts a constant background from EELS spectra by averaging the signal
    in the specified regions.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    x : tuple
        The range in the x-direction to consider for background calculation.
    y : tuple
        The range in the y-direction to consider for background calculation.

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.
    """

    # Averaged region: rectangular spatial ROI
    
    if x[0] >= x[1] or y[0] >= y[1] or np.any(np.array(x) < 0) or np.any(np.array(y) < 0):
        raise ValueError("\nIn background_constant: invalid ROI dimensions\n")
    
    roi = hs.roi.RectangularROI(left=x[0], right=x[1], top=y[0], bottom=y[1])
    small_s = roi(s)

    # Average in roi = background
    background = np.mean(small_s.data, axis=(0, 1))
        
    return s - background

data_path = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data_triangle/4-STEM SI.dm4"
s = hs.load(data_path)[-1]
s.plot()
plt.show(block=False)
background_constant(s, (0, 10), (0, 10)).plot()
plt.show(block=False)

input("Press Enter to continue...")
