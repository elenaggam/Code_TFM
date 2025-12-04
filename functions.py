import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.interpolate import interp1d
import time

# Background substraction for EELS spectra
def check_roi_E(s, x, y, E=None):
    """
    Checks the validity of the input parameters for background removal.

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
    x_check : bool
        True if the x range is valid, False otherwise.
    y_check : bool
        True if the y range is valid, False otherwise.
    E_check : bool
        True if the energy range is valid, False otherwise.

    """

    # Area check
    x_check = True
    x_range = (float(np.round(s.axes_manager[0].axis[0], 2)), float(np.round(s.axes_manager[0].axis[-1], 2)))
    if x[0] >= x[1] or np.any(np.array(x) < 0) or x[1] > x_range[1] or x[0] < x_range[0]:
        x_check = False

    y_check = True
    y_range = (float(np.round(s.axes_manager[1].axis[0], 2)), float(np.round(s.axes_manager[1].axis[-1], 2)))
    if y[0] >= y[1] or np.any(np.array(y) < 0) or y[1] > y_range[1] or y[0] < y_range[0]:
        y_check = False

    # Energy range check
    E_check = True
    if E is not None:
        E_range = (float(np.round(s.axes_manager[2].axis[0], 2)), float(np.round(s.axes_manager[2].axis[-1], 2)))
        if E[0] >= E[1] or np.any(np.array(E) < E_range[0]) or E[1] > E_range[1] or E[0] < E_range[0]:
            E_check = False


    return x_check, y_check, E_check

def get_roi_E(s, x, y, E=None):
    """
    Selects the region of interest (ROI) and energy range for background removal.

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
        The selected region of interest and energy range.
    """

    x_c, y_c, E_c = check_roi_E(s, x, y, E)
    if not x_c:
        raise ValueError(f"\nIn get_roi_E: invalid x range\n")
    if not y_c:
        raise ValueError(f"\nIn get_roi_E: invalid y range\n")
    if E is not None and not E_c:
        raise ValueError(f"\nIn get_roi_E: invalid energy range\n")
    
    roi = hs.roi.RectangularROI(left=x[0], right=x[1], top=y[0], bottom=y[1])
    small_s = roi(s)

    if E is not None:
        small_s = small_s.isig[E[0]:E[1]]

    return small_s

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

    small_s = get_roi_E(s, x, y, E)
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


# Criteria to select roi for background removal: lowest intensity values
def get_avg_in_window(s, threshold, E=None):
    """
    Helper function to calculate the average of intensity values below a threshold.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    threshold : float
        The threshold value to set for negative intensities.
    E : tuple (optional)
        The energy range to consider for the calculation in eV.

    Returns:
    --------
    float
        The average of intensity values below the threshold.
    """

    if E is not None:
        E_range = (float(np.round(s.axes_manager[2].axis[0], 2)), float(np.round(s.axes_manager[2].axis[-1], 2)))
        if E[0] >= E[1] or np.any(np.array(E) < E_range[0]) or E[1] > E_range[1] or E[0] < E_range[0]:
            raise ValueError(f"\nIn get_avg_window: invalid energy range, not in {E_range}\n")
        s2 = s.isig[E[0]:E[1]]
    else:
        s2 = s 

    if np.any(s2.data < threshold):
        avg = np.mean(s2.data[s2.data < threshold])
        #print(f"\nIn get_avg_window: {np.sum(s2.data < threshold)} intensity values below a threshold of {threshold} counts found, applying correction, to be set as {avg:.3f}... \n")
        return avg
    else:
        return None
 
def best_avg_roi(s, nx, ny, threshold, E=None, x0=0, y0=0, xf=None, yf=None, steps_x=20, steps_y=20, delta_step_x=None, delta_step_y=None):
    """
    Helper function to calculate the average of intensity values below a threshold
    in a specified region of interest (ROI).

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    nx : float
        The range in the x-direction to consider for ROI.
    ny : float
        The range in the y-direction to consider for ROI.
    threshold : float
        The threshold value to set for negative intensities.
    E : tuple (optional)
        The energy range to consider for the calculation in eV.
    x0 : float
        The starting x-coordinate for ROI search (default is 0).
    y0 : float
        The starting y-coordinate for ROI search (default is 0).
    xf : float (optional)
        The ending x-coordinate for ROI search (default is None, meaning the maximum x-axis value).
    yf : float (optional)
        The ending y-coordinate for ROI search (default is None, meaning the maximum y-axis value).
    steps_x : int
        The number of steps in the x-direction for ROI search (default is 100).
    steps_y : int
        The number of steps in the y-direction for ROI search (default is 100).
    delta_step_x : float (optional)
        The step size for ROI search in the x direction.
    delta_step_y : float (optional)
        The step size for ROI search in the y direction.
    Returns:
    --------
    best_avg : float
        The lowest average of intensity values below the threshold found in the ROI search.
    x_best : tuple
        The x-coordinates of the best ROI found.
    y_best : tuple
        The y-coordinates of the best ROI found.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError(f"\nIn get_avg_roi: invalid ROI size, nx and ny must be positive\n")
    
    best_avg = np.max(s.data)  # Initialize with the maximum possible value
    x_best = None
    y_best = None

    if delta_step_x is None:
        if xf is None:
            xf = s.axes_manager[0].axis[-1]
        delta_step_x = (xf - x0 - nx) / steps_x

    if delta_step_y is None:
        if yf is None:
            yf = s.axes_manager[1].axis[-1]
        delta_step_y = (yf - y0 - ny) / steps_y

    print(f"\nIn get_avg_roi: Starting ROI search.\n------------\n% completed\ttime in iteration\ttime remaining")

    abs_start_time = time.time()
    perc = 10
    for i in range(steps_x):
        start_t= time.time()

        # roi dimensions in x
        x_start = x0 + i * delta_step_x
        x_end = x_start + nx
        if x_end > s.axes_manager[0].axis[-1]:
            print(f"In get_avg_roi: Breaking x loop at step {i}")
            break

        for j in range(steps_y):
            # roi dimensions in y
            y_start = y0 + j * delta_step_y
            y_end = y_start + ny
            if y_end > s.axes_manager[1].axis[-1]:
                print(f"\nIn get_avg_roi: Breaking y loop at step {j}\n")
                break
            
            # Check roi validity
            x_c, y_c, E_c = check_roi_E(s, (x_start, x_end), (y_start, y_end), E)
            if E_c is False:
                raise ValueError(f"\nIn get_avg_roi: invalid energy range\n")
            if not x_c or not y_c:
                print("Invalid roi at x: ", (x_start, x_end), " y: ", (y_start, y_end), ', skipping...\n')
                continue

            roi = hs.roi.RectangularROI(left=x_start, right=x_end, top=y_start, bottom=y_end)
            small_s = roi(s)

            avg = get_avg_in_window(small_s, threshold, E)
            if avg is not None and avg < best_avg: # Looking for the lowest average, as counts rise as we hit the nanoparticle
                best_avg = avg
                x_best = (x_start, x_end)
                y_best = (y_start, y_end)

        t = time.time()
        a = (i+1)*100/steps_x
        if a >= perc:
            if perc == 100:
                break
            remaining = (t - start_t)*(steps_x - i - 1)
            if remaining < 60:
                print(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining:.2f} s")
            else:
                print(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining/60:.2f} min")
            perc = round(a) + 10

    
    total_t = time.time() - abs_start_time
    print(f"{100}\t\t{t - start_t:.2f} s\t\t\tTotal runtime: {total_t:.2f} s")
    print("------------")

    return best_avg, x_best, y_best


# Correction: Positive intensity values or at least close to zero
def ic_zero(s, E=None):
    """
    Correction: Positive intensity values by setting all negative values to zero.
    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    E : tuple (optional)
        The energy range to consider for the correction in eV.
    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The corrected EELS spectrum.    
    """

    if E is not None:
        E_range = (float(np.round(s.axes_manager[2].axis[0], 2)), float(np.round(s.axes_manager[2].axis[-1], 2)))
        if E[0] >= E[1] or np.any(np.array(E) < E_range[0]) or E[1] > E_range[1] or E[0] < E_range[0]:
            raise ValueError(f"\nIn ic_zero: invalid energy range, not in {E_range}\n")
        s2 = s.isig[E[0]:E[1]]
    else:
        s2 = s
    if np.any(s2.data < 0):
        print(f"\nIn ic_zero: {np.sum(s2.data < 0)} negative intensity values found, min={np.min(s2.data):.0f}, applying correction...")
        s.data = 0  
    
    print("...correction completed.")
    return s

def ic_naive(s, E=None):
    """
    Correction: Positive intensity values by shifting the spectrum upwards.
    The minimum intensity value is set to zero.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    E : tuple (optional)
        The energy range to consider for the correction in eV.

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The corrected EELS spectrum.    
    """
    
    if E is not None:
        E_range = (float(np.round(s.axes_manager[2].axis[0], 2)), float(np.round(s.axes_manager[2].axis[-1], 2)))
        if E[0] >= E[1] or np.any(np.array(E) < E_range[0]) or E[1] > E_range[1] or E[0] < E_range[0]:
            raise ValueError(f"\nIn ic_naive: invalid energy range, not in {E_range}\n")
        s2 = s.isig[E[0]:E[1]]
    else:
        s2 = s
    if np.any(s2.data < 0):
        a = s2.data.min()
        print(f"\nIn ic_naive: {np.sum(s2.data < 0)} negative intensity values found, min={a:.0f}, applying correction...")
        s.data -= a    
    # tends to produce overestimation of the background, big shifts
    print("...correction completed.")
    return s

def ic_threshold(s, threshold=0, E=None):
    """
    Correction: Positive intensity values by setting negative values below a threshold to the threshold value.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    threshold : float
        The threshold value to set for negative intensities (default is 0).
        Number of counts.
        It must be adecuate in order not to lose relevant information of the spectrum, like the plasmon peak.
    E : tuple (optional)
        The energy range to consider for the correction in eV.

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The corrected EELS spectrum.    
    """
    
    if E is not None:
        E_range = (float(np.round(s.axes_manager[2].axis[0], 2)), float(np.round(s.axes_manager[2].axis[-1], 2)))
        if E[0] >= E[1] or np.any(np.array(E) < E_range[0]) or E[1] > E_range[1] or E[0] < E_range[0]:
            raise ValueError(f"\nIn ic_threshold: invalid energy range, not in {E_range}\n")
        s2 = s.isig[E[0]:E[1]]
    else:
        s2 = s 
    if np.any(s2.data < threshold):
        print(f"\nIn ic_threshold: {np.sum(s2.data < threshold)} intensity values below a threshold of {threshold} counts found, applying correction...")
        s2.data[s2.data < threshold] = threshold

        if E is not None:
            s.isig[E[0]:E[1]] = s2
        else:
            s = s2
    print("...correction completed.")
    return s

def ic_averaged(s, threshold=0, E=None):
    """
    Correction: Positive intensity values by setting negative values below a threshold to the average
    of these values.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    threshold : float
        The threshold value to set for negative intensities (default is 0).
        Number of counts.
        It must be adecuate in order not to lose relevant information of the spectrum, like the plasmon peak.
    E : tuple (optional)
        The energy range to consider for the correction in eV.

    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The corrected EELS spectrum.
    """

    print(f"\nIn ic_averaged: calculating average of intensity values below a threshold of {threshold} counts...")
    avg = get_avg_in_window(s, threshold, E)
    if avg is not None:
        print(avg)
        if avg < 0:
            s.data += abs(avg)
        else:
            s.data[s.data < threshold] = avg
    print("...correction completed.")
    return s

def intensity_correction(s, threshold=0, E=None, name='naive'):
    """
    Corrects intensity values in EELS spectra using the specified method.

    Parameters:
    -----------
    name : str
        The method to use for intensity correction. Options are:
        'naive' - shifts the spectrum upwards to set minimum to zero,
        'threshold' - sets negative values below a threshold to the threshold value,
        'averaged' - sets negative values below a threshold to their average value,
        'zero' - sets all negative values to zero.
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    threshold : float
        The threshold value to set for negative intensities (default is 0).
        Number of counts.
        It must be adecuate in order not to lose relevant information of the spectrum, like the plasmon peak.
    E : tuple (optional)
        The energy range to consider for the correction in eV.
    Returns:    
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The corrected EELS spectrum.
    """
    if name == 'zero':
        return ic_zero(s, E)
    
    if name == 'naive':
        return ic_naive(s, E)
    
    if name == 'threshold':
        return ic_threshold(s, threshold, E)

    if name == 'averaged':
        return ic_averaged(s, threshold, E)
    
    
    raise ValueError("\nIn intensity_correction: unknown method name\n")
    




