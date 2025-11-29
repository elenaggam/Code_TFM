import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.interpolate import interp1d


# Background substraction for EELS spectra
def background_first_step(s, x, y, E=None):
    """
    First step done for background removal in EELS spectra.
    Checks the validity of the input parameters.
    Selects roi and energy range.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    x : tuple
        The range in the x-direction to consider for background calculation.
    y : tuple
        The range in the y-direction to consider for background calculation.
    E : tuple (optional)
        The energy range to consider for background calculation in eV.

    Returns:
    --------
    averaged_pixels : np.ndarray
        The averaged spectrum over the selected region.
    small_s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The data of interest for background calculation.
    """

    # Area check
    if x[0] >= x[1] or y[0] >= y[1] or np.any(np.array(x) < 0) or np.any(np.array(y) < 0):
        raise ValueError("\nIn background_constant: invalid ROI dimensions\n")
    
    roi = hs.roi.RectangularROI(left=x[0], right=x[1], top=y[0], bottom=y[1])
    small_s = roi(s)

    # Energy range check
    if E is not None:
        if E[0] >= E[1] or E[0] < small_s.axes_manager[2].axis[0] or E[1] > small_s.axes_manager[2].axis[-1]:
            raise ValueError("\nIn background_polyfit: invalid energy range\n")
        small_s = small_s.isig[E[0]:E[1]]

    # Pixels are averaged to get a single spectrum called background
    averaged_pixels = np.mean(small_s.data, axis=(0,1))

    return averaged_pixels, small_s

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
        The energy range to consider for background calculation in eV.

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.
    """

    background, small_s = background_first_step(s, x, y, E)
    
    if E is not None:
        s.isig[E[0]:E[1]] -= background
    else:
        s.data -= background
        
    return s

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
        The energy range to consider for background calculation in eV.
    order : int
        The order of the polynomial interpolation (default is 1 for linear).

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.
    """

    averaged_pixels, small_s = background_first_step(s, x, y, E)
    
    # Polynomial fit to the background spectrum
    coeffs = np.polyfit(small_s.axes_manager[2].axis, averaged_pixels, order)
    background = np.polyval(coeffs, small_s.axes_manager[2].axis)
    
    if E is not None:
        s.isig[E[0]:E[1]] -= background
    else:
        s.data -= background

    return s

def background_interpolation(s, x, y, E=None):
    """
    Subtracts a background from EELS spectra by interpolating the signal.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    x : tuple
        The range in the x-direction to consider for background calculation.
    y : tuple
        The range in the y-direction to consider for background calculation.
    E : tuple (optional)
        The energy range to consider for background calculation in eV.
    order : int
        The order of the polynomial interpolation (default is 1 for linear).

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.
    """

    averaged_pixels, small_s = background_first_step(s, x, y, E)
    
    # Interpolation to the background spectrum
    f = interp1d(small_s.axes_manager[2].axis, averaged_pixels, kind='cubic')
    background = f(small_s.axes_manager[2].axis)

    if E is not None:
        s.isig[E[0]:E[1]] -= background
    else:
        s.data -= background

    return s

def background_removal(s, x, y, E=None, name='interpolation', order=None):
    """
    Subtracts background from EELS spectra using the specified method.

    Parameters:
    -----------
    name : str
        The method to use for background subtraction. Options are:
        'constant' - constant background subtraction,
        'polyfit' - polynomial fit background subtraction,
        'interpolation' - interpolation background subtraction.
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    x : tuple
        The range in the x-direction to consider for background calculation.
    y : tuple
        The range in the y-direction to consider for background calculation.
    E : tuple (optional)
        The energy range to consider for background calculation in eV.
    order : int (optional)
        The order of the polynomial interpolation (only for 'polyfit')

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.
    """

    if name == 'constant':
        return background_constant(s, x, y, E)
    
    if name == 'polyfit':
        if order is None or order<1:
            raise ValueError("\nIn background_subtraction: 'order' parameter must be specified correctly for 'polyfit' method\n")
        if order == 0:
            print("\nIn background_subtraction: 'order' parameter cannot be 0 for 'polyfit' method, switching to 'constant' method\n")
            return background_constant(s, x, y, E)
        return background_polyfit(s, x, y, E, order)

    if name == 'interpolation':
        return background_interpolation(s, x, y, E)
    
    raise ValueError("\nIn background_subtraction: unknown method name\n")





data_path_base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
data_path = data_path_base + 'Triangle/4-STEM SI.dm4'
s = hs.load(data_path)[-1]

background_removal(s, (0, 10), (0, 10), E=(0.,3.)).plot()
plt.show()


input("Press Enter to continue...")


