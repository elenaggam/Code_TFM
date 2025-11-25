import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs


# Background substraction for EELS spectra



def background_constant(s, x, y, E=None):
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
    E : tuple (optional)
        The energy range to consider for background calculation.

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.
    """

    # Area check
    if x[0] >= x[1] or y[0] >= y[1] or np.any(np.array(x) < 0) or np.any(np.array(y) < 0):
        raise ValueError("\nIn background_constant: invalid ROI dimensions\n")
    
    roi = hs.roi.RectangularROI(left=x[0], right=x[1], top=y[0], bottom=y[1])
    small_s = roi(s)


    # Energy range check
    if E is not None:
        if E[0] >= E[1]:
            raise ValueError("\nIn background_constant: invalid energy range\n")
        small_s = small_s.inav.slice(E[0], E[1], 'energy')


    # Average in roi = background
    background_spectra = np.mean(small_s.data, axis=(0, 1))
    background = np.mean(background_spectra)
        
    return s - background


def background_polyfit(s, x, y, E=None, order=1):
    """
    Subtracts a background from EELS spectra by interpolating the signal
    in the specified regions as a polynomial function.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    x : tuple
        The range in the x-direction to consider for background calculation.
    y : tuple
        The range in the y-direction to consider for background calculation.
    E : tuple (optional)
        The energy range to consider for background calculation.
    order : int
        The order of the polynomial interpolation (default is 1 for linear).

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.
    """

    # Order check
    if order < 0:
        raise ValueError("\nIn background_interpolation: order must be non-negative\n")
    if order==0:
        print("\nIn background_interpolation: order 0 is equivalent to background_constant\n")
        return background_constant(s, x, y, E)
    

    # Area check
    if x[0] >= x[1] or y[0] >= y[1] or np.any(np.array(x) < 0) or np.any(np.array(y) < 0):
        raise ValueError("\nIn background_interpolation: invalid ROI dimensions\n")
    
    # Interpolated region: rectangular spatial ROI
    roi = hs.roi.RectangularROI(left=x[0], right=x[1], top=y[0], bottom=y[1])
    small_s = roi(s)


    # Energy range check
    if E is not None:
        if E[0] >= E[1]:
            raise ValueError("\nIn background_interpolation: invalid energy range\n")
        small_s = small_s.inav.slice(E[0], E[1], 'energy')

    
    # Interpolation process
    # Pixels are averaged to get a single spectrum called background
    averaged_pixels = np.mean(small_s.data, axis=(0, 1))

    # Polynomial fit to the background spectrum
    coeffs = np.polyfit(small_s.axes_manager[0].axis, averaged_pixels, order)
    background = np.polyval(coeffs, s.axes_manager[0].axis)
        

    return s - background

data_path_base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
data_path = data_path_base + 'Triangle/4-STEM SI.dm4'
s = hs.load(data_path)[-1]

background_polyfit(s, (0, 10), (0, 10), order=5).plot()
plt.show(block=False)

input("Press Enter to continue...")
