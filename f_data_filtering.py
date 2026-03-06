import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.interpolate import interp1d
import time
import os
import cv2 as cv


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

def background_thesis(s, b):
    """
    Background removal method based on normalizing the background spectrum (thesis)
    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    b : np.ndarray
        The background spectrum to be removed.  
    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The background-subtracted EELS spectrum.

    """

        # divide each of the reduced spectra by the background, channel by channel, search minimum
    for i in range(s.data.shape[0]):  # x-axis
        for j in range(s.data.shape[1]):  # y-axis
            C_ij = s.data[i,j,:]/b

            # global minimum point and corresponding intensity values
            # I_min = np.min(C_ij)
            E_min = np.argmin(C_ij)  # int, index of the global minimum point of C_ij
            S_min = s.data[i,j,E_min]
            Ib_min = b[E_min]

            # normalise the background with respect to each one of the spectra
            # subtract the normalised background from each of the spectra
            s.data[i,j] -= b*S_min/Ib_min

    # E_min = None
    # I_min = np.max(s.data)  # initialize with maximum possible value
    
    # # divide each of the reduced spectra by the background, channel by channel, search minimum
    # for i in range(s.data.shape[0]):  # x-axis
    #     for j in range(s.data.shape[1]):  # y-axis
    #         I = s.data[i,j]/b
    #         if I.min() < I_min:
    #             I_min = I.min()
    #             E_min = np.argmin(I)  # int, index of the global minimum point of I

    # print(E_min, I_min)
    # if E_min is None:
    #     raise ValueError("\nIn back: oops, could not find minimum intensity value\n")
    
    # # global minimum point, corresponding values in s (in loop) and b
    # Ib_min = b[E_min] # float
    # print(Ib_min)

    # # normalise the background with respect to each one of the spectra
    # # subtract the normalised background from each of the spectra
    # for i in range(s.data.shape[0]):  # x-axis
    #     for j in range(s.data.shape[1]):  # y-axis
    #         s_min = s.data[i,j, E_min] # float, intensity value at E_min of each spectrum
    #         s.data[i,j] = s.data[i,j] - b*s_min/Ib_min

    return s

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
def get_avg_in_window(s, threshold=1e7, E=None):
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
 
def best_avg_roi(directory, s, nx, ny, threshold=1e7, E=None, x0=0, y0=0, xf=None, yf=None, steps_x=20, steps_y=20, delta_step_x=None, delta_step_y=None):
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
        raise ValueError("\nIn get_avg_roi: invalid ROI size, nx and ny must be positive\n")
    
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
        
    outp = directory + "logs/"
    if not os.path.exists(outp):
        os.makedirs(outp)
    file_time = open(outp+f'{nx}_{ny}_best_avg_roi_time.txt', 'w')
    file_time.write("\nIn get_avg_roi: Starting ROI search.\n------------\n% completed\ttime in iteration\ttime remaining\n")

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
                raise ValueError("\nIn get_avg_roi: invalid energy range\n")
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
            if remaining > 3600:
                file_time.write(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining/3600:.2f} h\n")
            if remaining < 60:
                file_time.write(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining:.2f} s\n")
            else:
                file_time.write(f"{perc}\t\t{t - start_t:.2f} s\t\t\t{remaining/60:.2f} min\n")
            perc = round(a) + 10


    total_t = time.time() - abs_start_time
    file_time.write(f"{100}\t\t{t - start_t:.2f} s\t\t\tTotal runtime: {total_t:.2f} s\n")
    file_time.close()
    

    file = open(outp+f'{nx}_{ny}_best_avg_roi.txt', 'w')
    file.write('x_start\tx_end\ny_start\ty_end\tbest_avg\truntime(s)\tx_steps\ty_steps\tdelta_step_x\tdelta_step_y\n')
    file.write(f'{x_best[0]}\t{x_best[1]}\n{y_best[0]}\t{y_best[1]}\t{best_avg}\t{total_t:.2f}\t{steps_x}\t{steps_y}\t{delta_step_x:.2f}\t{delta_step_y:.2f}')
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
def background_removal(directory, s, nx=20, ny=20, x0=0, y0=0, xf=None, yf=None, steps_x=20, steps_y=20, delta_step_x=None, delta_step_y=None, E=None, name_bg='interpolation', name_ic='naive', threshold=0):
    '''
    Complete background removal from EELS spectra by calculating the background and correcting negative intensity values.
    '''
    t_init = time.time()
    # print("\nStarting background removal...\n")
    s.align_zero_loss_peak()
    # print("Zero-loss peak aligned")

    x, y = best_avg_roi(directory, s, nx, ny, threshold=threshold, E=E, x0=x0, y0=y0, xf=xf, yf=yf, steps_x=steps_x, steps_y=steps_y, delta_step_x=delta_step_x, delta_step_y=delta_step_y)
    # print(f"Best ROI found at x: {x} y: {y}")

    if name_bg not in ['constant', 'interpolation', 'thesis']:
        print("\nIn background_removal: unknown method name for background subtraction, using 'interpolation' instead\n")
        name_bg = 'interpolation'
    
    if name_bg == 'thesis':
        b = background_interpolation(s, x, y, E)
        s = background_thesis(s, b)
    else:
        s = s- get_background(s, x, y, E, name_bg)

        if name_ic not in ['naive', 'threshold', 'averaged', 'zero']:
            print("\nIn background_removal: unknown method name for intensity correction, using 'naive' instead\n")
            name_ic = 'naive'
    
        s = intensity_correction(s, threshold, E, name_ic)

    s.save(directory+name_bg+'_'+name_ic+'_background_removed.hspy')
    total_time = time.time() - t_init
    outp = directory + "logs/"
    if not os.path.exists(outp):
        os.makedirs(outp)
    file_time = open(outp+f'{name_bg}_{name_ic}_background_removal_time.txt', 'w')
    file_time.write(f"\nBackground removal completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes).\n")
    file_time.close()
    return 



# PCA, NNMF: rebuilding the spectra with the most relevant components to further reduce noise and enhance signal
def components_reduction(directory, s, method='sklearn_pca', n_components=None,  max_points=40, save_summary=True, save_components=True, plot_variance_ratio=True):
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
    save_summary : bool
        Whether to save a summary table of the explained variance and explained variance ratio for each component (default is True).
    save_components : bool
        Whether to save each component as a separate file (default is True).
    plot_variance_ratio : bool
        Whether to plot the explained variance ratio for each component (default is True).

    Returns:
    --------
    None
    '''

    if method not in ['sklearn_pca', 'sklearn_nnmf']:
        print("\nIn components_reduction: unknown method name, using 'sklearn_pca' instead\n")
        method = 'sklearn_pca'

    s.decomposition(method) 

    # save summary in a text file
    if save_summary:
        lr = s.learning_results
        n_components = lr.loadings.shape[1]
        explained_variance = lr.explained_variance
        explained_variance_ratio = lr.explained_variance_ratio
        outp = directory + "logs/"
        if not os.path.exists(outp):
            os.makedirs(outp)
        with open(outp+f"{method}_summary_table.txt", "w") as f:
            f.write("Component\tExplainedVariance\tExplainedVarianceRatio\n")
            for i in range(n_components):
                f.write(f"{i+1}\t{explained_variance[i]}\t{explained_variance_ratio[i]}\n")

    # plot explained variance ratio    
    if plot_variance_ratio:
        s.plot_explained_variance_ratio(n=max_points, vline=True)
        plt.savefig(directory+f"{method}_explained_variance_ratio.png", dpi=300, bbox_inches="tight")
    
    a = s.estimate_elbow_position(explained_variance_ratio=None, log=True, max_points=max_points)

    # save new signal with the most relevant components
    sc = s.get_decomposition_model(a)
    sc.save(directory+f"{method}_{a}_components.hspy")

    # save each component as a separate file
    if save_components:
        # factors = s.get_decomposition_factors().as_signal1D(1)
        # loadings = s.get_decomposition_loadings()
        dir_components = directory + f"{method}_components/"

        if not os.path.exists(dir_components):
            os.makedirs(dir_components)

        for i in range(a+5): # save a few more components than the elbow point, to be sure not to lose relevant information
            # component = np.einsum('ij,k->ijk', loadings.data[:,i], factors.data[i,:])
            # component_signal = hs.signals.Signal2D(component)
            # component_signal.axes_manager = factors.axes_manager.deepcopy()
            # component_signal.save(dir_components+f"{i+1}.hspy")

            _ = sc.plot_decomposition_loadings([i],title='')
            plt.savefig(dir_components+f"Componente_{i+1}.png",dpi=600, bbox_inches='tight')
    


    return



# binary mask to select the area outside of the nanoparticle
def otsu_mask(s, directory):
    '''
    Creates a binary mask using Otsu's thresholding method to select the area outside of the nanoparticle in the EELS spectrum.
    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    directory : str
        The directory where the results will be saved.
    Returns:
    --------
    None
    '''

    # Sum the energy channels to get a 2D array (flat image)
    image = s.sum(axis=2).data
    # Normalize the image to the range [0, 255] and convert to uint8 (8-bit unsigned integer to be compatible with OpenCV functions)
    image_norm = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    img_8bit = image_norm.astype(np.uint8)

    # Otsu's thresholding to get a binary mask (values of 0 inside and 1 outside)
    _, mask = cv.threshold(img_8bit, 0, 1, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    # Save the binary mask as a text file with integer values (0 and 1) with the 2D image's size
    np.savetxt(directory+"otsu_mask.txt", mask, fmt="%d")
    histogram, bin_edges = np.histogram(img_8bit.ravel(), bins=256)
    np.savetxt(directory+"otsu_histogram.txt", np.column_stack((bin_edges[:-1], histogram)), fmt="%d")

    # Plot the images and the histogram
    plt.figure(figsize=(15, 5))

    # original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_8bit, cmap='gray', aspect='equal')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # histogram
    plt.subplot(1, 3, 2)
    plt.hist(img_8bit.ravel(), bins=256, color='blue')
    plt.title('Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Pixel count')

    # masked image
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray', aspect='equal')
    plt.title("Otsu's Thresholding")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(directory+"otsu_thresholding.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # # Otsu's thresholding after Gaussian filtering (from OpenCV documentation, to reduce noise and improve the thresholding result)
    # blur = cv.GaussianBlur(img,(5,5),0)
    # ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # mask3 = cv.inRange(th3)

    return 

def apply_mask(s, directory, mask_file="otsu_mask.txt"):
    '''
    Applies a binary mask to the original EELS spectrum to select the area outside of the nanoparticle.
    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    directory : str
        The directory where the results will be saved.
    mask_file : str
        The name of the text file containing the binary mask, in the same directory.
    Returns:
    --------
    hs.signals.EELSSpectrum or hs.signals.Signal1D
        The masked EELS spectrum, with values inside the nanoparticle set to zero.
    '''

    # Load the binary mask from the text file
    mask = np.loadtxt(directory+mask_file, dtype=int)

    # Apply the mask to the original 3D EELS spectrum (set values inside the nanoparticle to zero)
    masked_data = s.data * mask[:,:,np.newaxis]  # Broadcasting the mask to match the shape of the data

    # Create a new EELSSpectrum with the masked data
    masked_spectrum = hs.signals.EELSSpectrum(masked_data, axes_manager=s.axes_manager.deepcopy())

    return masked_spectrum



# calculation of the dielectric function from the EELS spectrum using Kramers-Kronig analysis
def dielectric_function(s, directory):
    '''
    Calculates the dielectric function from the EELS spectrum using Kramers-Kronig analysis.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    directory : str
        The directory where the results will be saved.

    Returns:
    --------
    None
    '''

    thickness = s.estimate_thickness(threshold=None, zlp=None, density=None, mean_free_path=None)
    diel = s.kramers_kronig_analysis(zlp=None, iterations=2, n=None, t=thickness, delta=0.5, full_output=False)
    diel.save(directory+"dielectric_function.hspy")
    diel.plot()
    plt.show()
    return

dir = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/Triangle/"
s = hs.load(dir+'4-STEM SI.hspy') # Load your EELS spectrum here
s2 = apply_mask(s, dir)
dielectric_function(s2, dir)

