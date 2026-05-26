import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.interpolate import interp1d
import time
import os
import cv2 as cv
from exspy.signals import EELSSpectrum
import datetime
import f_dielectric as df



# Background calculation for EELS spectra
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
        if E[0] >= E[1] or E[1] > E_range[1] or E[0] < E_range[0]:
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
        raise ValueError("\nIn get_roi_E: invalid x range\n")
    if not y_c:
        raise ValueError("\nIn get_roi_E: invalid y range\n")
    if E is not None and not E_c:
        raise ValueError("\nIn get_roi_E: invalid energy range\n")
    
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

    return np.mean(small_s.data, axis=(0,1)), small_s

def background_constant(s, x, y, E=None):
    """
    Calculates a constant background from EELS spectra by averaging the signal
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
        The background calculated from the EELS spectrum.
    """

    background, small_s = background_first_step(s, x, y, E)
        
    return background

def background_interpolation(s, x, y, E=None):
    """
    Calculates background from EELS spectra by interpolating the signal.

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
        The background
    """

    averaged_pixels, small_s = background_first_step(s, x, y, E)
    
    # Interpolation to the background spectrum
    f = interp1d(small_s.axes_manager[2].axis, averaged_pixels, kind='cubic')

    return f(small_s.axes_manager[2].axis)

def background_thesis(s, b, rebin_factor=3, num_channels=20):
    """
    Método de eliminación de fondo basado en la normalización 
    del espectro de fondo según la tesis.
    """
    s_ext = s.deepcopy()
    
    # as arrays for faster execution
    S_original = np.asarray(s_ext.data)
    B_original = np.asarray(b.data) if hasattr(b, 'data') else np.asarray(b)
    
    def rebin_1d(arr, factor): # for b
        shape = arr.shape[0] // factor, factor
        return arr[:shape[0] * factor].reshape(shape).mean(-1)
    
    def rebin_3d(arr, factor): # for s
        Ny, Nx, Nc = arr.shape
        Nc_new = Nc // factor
        return arr[:, :, :Nc_new * factor].reshape(Ny, Nx, Nc_new, factor).mean(-1)

    
    S_red = rebin_3d(S_original, rebin_factor)[:, :, :num_channels]
    B_red = rebin_1d(B_original, rebin_factor)[:num_channels]
    
    for i in range(S_original.shape[0]): # x
        for j in range(S_original.shape[1]): # y 
            
            C_ij = S_red[i, j, :] / B_red[:]
            
            idx = np.argmin(C_ij) # index for Emin
            S_min_ij = S_red[i, j, idx]
            IBmin = B_red[idx]
 
            B_norm = (S_min_ij/IBmin)*B_original
            
            s_ext.data[i, j, :] = S_original[i, j, :] - B_norm

    return s_ext


def get_background(s, x, y, E=None, name='interpolation'):
    """
    Calculates background from EELS spectra using the specified method.

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
        The background calculated from the EELS spectrum.
    """

    if name == 'constant':
        return background_constant(s, x, y, E)

    if name == 'interpolation':
        return background_interpolation(s, x, y, E)
    
    if name == 'thesis':
        b = background_interpolation(s, x, y, E)
        return background_thesis(s, b)
    
    raise ValueError("\nIn background_subtraction: unknown method name\n")



# Criteria to select roi for background removal: lowest intensity values
def get_avg_in_window(s, threshold=1e9, E=None):
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
 
def best_avg_roi(directory, s, nx, ny, threshold=1e9, E=None, x0=0, y0=0, xf=None, yf=None, steps_x=None, steps_y=None, delta_step_x=None, delta_step_y=None):
    """
    Helper function to calculate the average of intensity values below a threshold
    in a specified region of interest (ROI).

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    nx : float (nm)
        The range in the x-direction to consider for ROI.
    ny : float (nm)
        The range in the y-direction to consider for ROI.
    threshold : float
        The threshold value to set for negative intensities.
    E : tuple (optional)
        The energy range to consider for the calculation in eV.
    x0 : float (nm)
        The starting x-coordinate for ROI search (default is 0).
    y0 : float (nm)
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
    x_best : tuple
        The x-coordinates of the best ROI found.
    y_best : tuple
        The y-coordinates of the best ROI found.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError("\nIn get_avg_roi: invalid ROI size, nx and ny must be positive\n")
    
    best_avg = np.min(s.data)  # Initialize with the minimum possible value
    x_best = None
    y_best = None

    if xf is None:
        xf = s.axes_manager[0].axis[-1]
    if yf is None:
        yf = s.axes_manager[1].axis[-1]

    if steps_x is None:
        steps_x = int((xf - x0)/nx)  # default number of steps to cover the whole area with half-overlapping ROIs
    if steps_y is None:
        steps_y = int((yf - y0)/ny)  # default number of steps to cover the whole area with half-overlapping ROIs
    
    if delta_step_x is None:
        delta_step_x = (xf - x0 - nx)/float(steps_x)
        print(delta_step_x)

    if delta_step_y is None:
        delta_step_y = (yf - y0 - ny)/float(steps_y)
        
    
    # file_time.write("\nIn get_avg_roi: Starting ROI search.\n------------\n% completed\ttime in iteration\ttime remaining\n")

    abs_start_time = time.time()
    for i in range(steps_x):
        # start_t= time.time()

        # roi dimensions in x
        x_start = x0 + i * delta_step_x
        x_end = x_start + nx
        #print(i, x_start, x_end)
        if x_end > s.axes_manager[0].axis[-1]:
            print(f"In get_avg_roi: Breaking x loop at step {i}")
            break

        for j in range(steps_y):
            # roi dimensions in y
            y_start = y0 + j * delta_step_y
            y_end = y_start + ny
            
            #print(y_start, y_end)
            if y_end > s.axes_manager[1].axis[-1]:
                print(f"\nIn get_avg_roi: Breaking y loop at step {j}\n")
                break
            
            # Check roi validity
            x_c, y_c, E_c = check_roi_E(s, (x_start, x_end), (y_start, y_end), E)
            if E_c is False:
                raise ValueError("\nIn get_avg_roi: invalid energy range\n")
            if not x_c or not y_c:
                print("Invalid roi at x: ", (x_start, x_end), " y: ", (y_start, y_end), ', skipping...\n')
                continue

            small_s = get_roi_E(s, (x_start, x_end), (y_start, y_end), E)
            avg = get_avg_in_window(small_s, threshold)
            if avg is not None and avg > best_avg: # Looking for the highest average, as counts rise as we hit the nanoparticle
                best_avg = avg
                x_best = (x_start, x_end)
                y_best = (y_start, y_end)

        # t = time.time()
        # a = (i+1)*100/steps_x
        # if a >= perc:
        #     if perc == 100:
        #         break
        #     remaining = (t - start_t)*(steps_x - i - 1)
        #     if remaining > 3600:
        #         file_time.write(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining/3600:.2f} h\n")
        #     if remaining < 60:
        #         file_time.write(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining:.2f} s\n")
        #     else:
        #         file_time.write(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining/60:.2f} min\n")
        #     perc = round(a) + 10


    total_t = time.time() - abs_start_time
    file = open(directory+f'best_avg_roi.txt', 'a')
    if os.stat(directory+f'best_avg_roi.txt').st_size == 0:
        file.write(f't(s)\t{total_t:.2f}\nx0\t{x0}\nxf\t{xf}\ny0\t{y0}\nyf\t{yf}\nnx\t{nx}\nny\t{ny}\navg\t{best_avg:.3f}\nx_steps\t{steps_x}\ny_steps\t{steps_y}\ndelta_x\t{delta_step_x:.2f}\ndelta_y\t{delta_step_y:.2f}\nthreshold(eV)\t{threshold:.2f}\n')
    if E:
        file.write(f'Energy range: {E[0]}-{E[1]} eV\n')
    file.write(f'x_best\t{x_best[0]}\ny_best\t{y_best[0]}\n')
    file.close()

    return x_best, y_best



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
    
    # print("...correction completed.")
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
        a = s2.data.min() # !!! background bien!!!
        print(f"\nIn ic_naive: {np.sum(s2.data < 0)} negative intensity values found, min={a:.0f}, applying correction...")
        s.data -= a    
    # tends to produce overestimation of the background, big shifts
    # print("...correction completed.")
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
    # print("...correction completed.")
    return s

def ic_averaged(s, threshold=1e6, E=None):
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

    # print(f"\nIn ic_averaged: calculating average of intensity values below a threshold of {threshold} counts...")
    avg = get_avg_in_window(s, threshold, E)
    if avg is not None:
        # print(avg)
        if avg < 0:
            s.data += abs(avg)
        else:
            s.data[s.data < threshold] = avg
    # print("...correction completed.")
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
    


# complete background removal: background area selection, calculation + correction of negative values
def background_removal(directory, s, nx=15, ny=15, x0=0, y0=0, xf=None, yf=None, steps_x=None, steps_y=None, delta_step_x=None, delta_step_y=None, E=None, name_bg='interpolation', name_ic='naive', threshold=1e8, dm4_filename=None):
    '''
    Complete background removal from EELS spectra by calculating the background and correcting negative intensity values.
    '''
    s.isig[0.0:]
    # Create output directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    logs = directory + 'logs/'
    if not os.path.exists(logs):
        os.makedirs(logs)


    x, y = best_avg_roi(directory, s, nx, ny, threshold=s.data.max(), E=E, x0=x0, y0=y0, xf=xf, yf=yf, steps_x=steps_x, steps_y=steps_y, delta_step_x=delta_step_x, delta_step_y=delta_step_y)
    print(f"Best ROI found at x: {x} y: {y}")

    if name_bg not in ['constant', 'interpolation', 'thesis']:
        print("\nIn background_removal: unknown method name for background subtraction, using 'interpolation' instead\n")
        name_bg = 'interpolation'
    
    if name_bg == 'thesis':
        b = background_interpolation(s, x, y)
        s = background_thesis(s, b)
    else:
        b = get_background(s, x, y, name=name_bg)
        s = s - b

    if name_ic not in ['naive', 'threshold', 'averaged', 'zero']:
        print("\nIn background_removal: unknown method name for intensity correction, using 'naive' instead\n")
        name_ic = 'naive'

    s = intensity_correction(s, threshold, E, name_ic)

    s.save(directory+'Data.hspy', overwrite=True)
    print("Background removed and intensity corrected, data saved")
    # np.savetxt(directory+'Background.hspy', b)
    return 



# PCA, NNMF: rebuilding the spectra with the most relevant components to further reduce noise and enhance signal
def components_reduction(directory, s, method='sklearn_pca', n_components=None,  max_points=40, save_components=False, add = 0, plot_components=True, cmap='coolwarm'):
    '''
    Reduces the number of components in the EELS spectra using PCA or NNMF, by rebuilding the spectra with the most relevant components.

    Parameters:
    -----------
    directory : str
        The directory where the results will be saved.
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.  
    method : str
        The method to use for dimensionality reduction. Options are:
        'sklearn_pca' - PCA using scikit-learn implementation,
        'sklearn_nnmf' - NNMF using scikit-learn implementation.
    n_components : int (optional)
        The number of components to keep (default is None, meaning it will be determined by the elbow method).
    max_points : int
        The maximum number of components to consider for the elbow method (default is 40).  
    save_components : bool
        Whether to save each component as a separate file (default is False).
    plot_components : bool
        Whether to plot each component (default is True).

    Returns:
    --------
    None
    '''

    if method not in ['sklearn_pca', 'NMF']:
        print("\nIn components_reduction: unknown method name, using 'sklearn_pca' instead\n")
        method = 'sklearn_pca'

    if not os.path.exists(directory):
        os.makedirs(directory)

    if method == 'NMF':     
        s.decomposition(algorithm=method, output_dimension=n_components, max_iter=n_components*100000,init='nndsvd')
    
    else: #pca
        outp = directory + "logs/"
        if not os.path.exists(outp):
            os.makedirs(outp)
        s.decomposition(algorithm=method) 

        lr = s.learning_results
        N_comp = lr.loadings.shape[1]
        explained_variance = lr.explained_variance
        explained_variance_ratio = lr.explained_variance_ratio
        with open(outp+"summary_table.txt", "w") as f:
            f.write("Component\tExplainedVariance\tExplainedVarianceRatio\n")
            for i in range(N_comp):
                f.write(f"{i+1}\t{explained_variance[i]}\t{explained_variance_ratio[i]}\n")
        
        s.plot_explained_variance_ratio(n=max_points, vline=True)
        plt.savefig(directory+"logs/explained_variance_ratio.png", dpi=300, bbox_inches="tight")
        plt.close()

    
    if n_components is not None:
        a = n_components
    else:
        a = s.estimate_elbow_position(explained_variance_ratio=None, log=True, max_points=max_points) + add

    # save new signal with the most relevant components
    sc = s.get_decomposition_model(a)
        
    if save_components:
        sc.save(directory+f"{a}_Data.hspy", overwrite=True)

    # save each component as a separate file
    if save_components or plot_components:
        dir_components = directory + "Component_"

        if plot_components:
            _ = sc.plot_decomposition_factors(a, title='', comp_label="")
            plt.savefig(dir_components+"factors.png",dpi=600, bbox_inches='tight')
            plt.close()
        for i in range(a): 
            if save_components:
                s.get_decomposition_model(components=[i]).save(dir_components+f"{i+1}.hspy", overwrite=True)
            if plot_components:
                _ = sc.plot_decomposition_loadings([i], title='', cmap=cmap, norm='log')
                plt.savefig(dir_components+f"{i+1}.png",dpi=600, bbox_inches='tight')
                plt.close()
                _ = sc.plot_decomposition_factors([i], title='')
                plt.savefig(dir_components+f"{i+1}_factor.png",dpi=600, bbox_inches='tight')
                plt.close()
                print(f'Component {i+1} saved.')
    
    return


def components_rebuild(directory, n_list, name_out='rebuilt'):
    
    
    for i in range(len(n_list)):
        if i==0:
            s = hs.load(directory+f"components/{n_list[i]}.hspy")
        else:
            s.data += hs.load(directory+f"components/{n_list[i]}.hspy").data
    s.metadata.General.title = "New Spectrum"
    s.save(directory+name_out+'.hspy', overwrite=True)  
    return s

def components_rebuild_file(comp_file, file_line, data_dir, output_name='rebuilt', all=False):
    

    comp = []
    with open(comp_file) as f:
        for i, line in enumerate(f):
            if i == file_line:  
                row = [int(x) for x in line.split()]
                comp.append(row)
                break
    # if we want to take intermediate components that are not in the file
    if all:
        new_list = [j+1 for j in range(comp[-1])]
        components_rebuild(data_dir, new_list, output_name)
    # just the specified components
    else:
        components_rebuild(data_dir, comp, output_name)
           
                
    return
        









# full data filtering process: ZLP alignment, background removal, intensity correction, mask application and dimensionality reduction

def dm4_pre_treatment(base, dm4, background):
    '''
    Given a directory containing the .dm4 files, aligns the ZLP and substracts the background using the desired method
    in the same directory with the suffixes "Aligned.hspy" and creates a background folder, respectively.
    '''
    file_idx = int(dm4[0])
    s = hs.load(base + dm4)[-1]

    back_folder = base + f'{file_idx}_' + background['name_bg'] + '_' + background['name_ic'] + '_BR/'

    if not os.path.exists(base + f'{file_idx}-Aligned.hspy'):
        s.align_zero_loss_peak()
        s.save(base + f'{file_idx}-Aligned.hspy', overwrite=True)
        del s
    print("Zero-loss peak aligned")

    s = hs.load(base + f'{file_idx}-Aligned.hspy')
    if background["done"] == False:
        background_removal(back_folder, s, name_bg=background['name_bg'], name_ic=background['name_ic'])

    print("Background removed and intensity corrected, data saved")
           
    return

def mask_process(base, dm4, mask):
    if not mask["done"]:
        data_scan = hs.load(os.path.join(base, dm4))[1]

        file_idx = int(dm4[0])
        dir_mask = base + f'{file_idx}-mask/' #masks folder, as they dont depend on the background method

        if not os.path.exists(dir_mask):
            os.makedirs(dir_mask)

        if mask["merge"]:
            df.get_mask(data_scan, dir_mask, method='otsu')
            df.get_mask(data_scan, dir_mask, method='adaptive')
            df.merge_masks(dir_mask)
        else:
            df.get_mask(data_scan, dir_mask, method=mask["method"])
        del data_scan
    return

def dimensionality_reduction(base, dim_reduction, E_slice=None, add = 0):
    s = hs.load(base+"Data.hspy")
    if E_slice is not None:
        s = s.isig[E_slice[0]:E_slice[1]]
    components_reduction(base + dim_reduction['folder_name'], s, method=dim_reduction['method'], n_components=dim_reduction['n_components'],  plot_components=dim_reduction['plot_components'], save_components=dim_reduction['save_components'], add=add)
    return

def data_treatment(base, dm4, background, mask, dim_reduction, E_slice=None, add = 0):
    file_idx = int(dm4[0])
    dm4_pre_treatment(base, dm4, background)
    mask_process(base, dm4, mask)
    dimensionality_reduction(base + f'{file_idx}_' + background['name_bg'] + '_' + background['name_ic'] + '_BR/', dim_reduction, E_slice=E_slice, add = add)
    return

def full_data_treatment(data_path_base, background, mask, dim_reduction, E_slice = None, add = 0):
   
    # get dm4 files in the directory data
    dm4_files = [f for f in os.listdir(data_path_base) if f.endswith('.dm4')]
    
    for dm4 in dm4_files:
        print(f'Processing {dm4}...')
        # align zlp, background, mask and dimensionality reduction
        data_treatment(data_path_base, dm4, background, mask, dim_reduction, E_slice=E_slice, add = add)

        print(f'Finished processing {dm4} at {datetime.datetime.now()}.\n')
    print(f'Finished processing all files in {data_path_base} at {datetime.datetime.now()}.\n')
    return