import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.interpolate import interp1d
import time
import os
import cv2 as cv
from exspy.signals import EELSSpectrum
import datetime
import f_data_filtering as f

# binary mask to select the area outside of the nanoparticle
def get_mask(s, directory, method='otsu'):
    '''
    Creates a binary mask using the specified method to select the area outside of the nanoparticle in the EELS spectrum.
    Parameters:
    -----------
    s : scan
    directory : str
        The directory where the results will be saved.
    method : str
        The method to use for creating the binary mask ('otsu' or 'adaptive').
    Returns:
    --------
    None
    '''

    # Normalize the image to the range [0, 255] and convert to uint8 (8-bit unsigned integer to be compatible with OpenCV functions)
    image = s.data
    image_norm = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    img_8bit = image_norm.astype(np.uint8)

    # Otsu's thresholding to get a binary mask (values of 0 inside and 1 outside)
    if method not in ['otsu', 'adaptive']:
        print("\nIn get_mask: unknown method name, using 'otsu' instead\n")
        method = 'otsu'

    if method == 'otsu':
        _, mask = cv.threshold(img_8bit, 0, 1, cv.THRESH_BINARY+cv.THRESH_OTSU)

    elif method == 'adaptive':
        mask = cv.adaptiveThreshold(img_8bit, 1, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv.THRESH_BINARY, blockSize=99, C=-3)
    mask = 1 - mask  # Invert the mask to have 1 outside the nanoparticle and 0 inside

    # Save the binary mask as a text file with integer values (0 and 1) with the 2D image's size
    np.savetxt(directory+method+'.txt', mask, fmt="%d")
    histogram, bin_edges = np.histogram(img_8bit.ravel(), bins=256)
    np.savetxt(directory+'histogram_'+method+'.txt', np.column_stack((bin_edges[:-1], histogram)), fmt="%d")

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
    plt.title(f"{method.capitalize()}'s Thresholding")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(directory+method+'_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # # Otsu's thresholding after Gaussian filtering (from OpenCV documentation, to reduce noise and improve the thresholding result)
    # blur = cv.GaussianBlur(img,(5,5),0)
    # ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # mask3 = cv.inRange(th3)

    return 

def apply_mask(s, mask_file):
    '''
    Applies a binary mask to the original EELS spectrum to select the area outside of the nanoparticle.
    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    mask_file : str
        The name of the text file containing the binary mask
    Returns:
    --------
    None
    '''

    # Load the binary mask from the text file
    mask = np.loadtxt(mask_file, dtype=int)

    # Apply the mask to the original 3D EELS spectrum (set values inside the nanoparticle to zero)
    for k in range(s.data.shape[2]):  # iterate over energy channels
        s.data[:,:,k] *=(1-mask)  # set values inside the nanoparticle to zero 

    return 

def merge_masks(directory, mask_files=["otsu.txt", "adaptive.txt"]):
    '''
    Merges two binary masks (from Otsu's and adaptive thresholding) to create a more accurate mask of the area outside of the nanoparticle.
    Parameters:
    -----------
    directory : str
        The directory where the results will be saved.
    mask_files : list of str
        The names of the text files containing the binary masks to be merged, in the same directory (default is ["otsu_mask.txt", "adaptive_mask.txt"]).
    Returns:
    --------
    None
    '''

    if not os.path.exists(directory+mask_files[0]) or not os.path.exists(directory+mask_files[1]):
        raise FileNotFoundError("\nIn merge_masks: one or both mask files not found in the directory\n")
    
    otsu = np.loadtxt(directory+mask_files[0], dtype=int)
    adaptive = np.loadtxt(directory+mask_files[1], dtype=int)
    mask = (1-otsu) + (1-adaptive)
    mask = np.clip(mask, 0, 1)
    mask = 1 - mask  # Invert the mask to have 1 outside the nanoparticle and 0 inside

    np.savetxt(directory+"mask_merged.txt", mask, fmt="%d")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(otsu, cmap='gray', aspect='equal')
    plt.title("Otsu's Thresholding Mask")
    plt.subplot(1, 3, 2)
    plt.imshow(adaptive, cmap='gray', aspect='equal')
    plt.title("Adaptive Thresholding Mask")
    # masked image
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray', aspect='equal')
    plt.title(f"Merged mask")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(directory+"mask_merged_results.png", dpi=300, bbox_inches='tight')
    plt.close()


# calculation of the dielectric function from the EELS spectrum using Kramers-Kronig analysis
def dielectric_function(s, zlp, dir_mask):
    '''
    Calculates the dielectric function from the EELS spectrum using Kramers-Kronig analysis.

    Parameters:
    -----------
    s : hs.signals.EELSSpectrum or hs.signals.Signal1D
        The input (EELS) spectrum.
    directory : str
        The directory where the results will be saved.
    dir_mask : str
        The directory where the binary mask is saved.
    dir_zlp : str
        The directory where the ZLP spectrum is saved.
    Returns:
    --------
    None
    '''
    thickness = hs.load('Relative thickness.dm4').data
    t = s.isig[0].copy() 
    
    # 3. Inyectar los datos del espesor en ese clon
    # Al ser s.isig[0], ya tiene dimensiones (422, 321) y los ejes perfectos
    t.data = thickness
    
    # Limpieza de seguridad
    t.data[t.data <= 0] = 1e-9
    
    # Cambiar el nombre para que no se confunda con EELS
    t.metadata.General.title = "Thickness Map"

    apply_mask(s, dir_mask)

    diel = s.kramers_kronig_analysis(zlp=zlp, iterations=2, n=None, t=t, delta=0.5, full_output=False)
    diel.save("dielectric_function.hspy", overwrite=True)
    diel.plot()
    plt.show()
    
    diel.real.plot()
    plt.show()
    diel.imag.plot()
    plt.show()
    
    return



def match_axis(s, zlp):
    s_axis = s.axes_manager[-1].axis
    zlp_axis = zlp.axes_manager[-1].axis

    low = max(s_axis.min(), zlp_axis.min())
    high = min(s_axis.max(), zlp_axis.max())

    # recortar SI (solo energía, NO navegación)
    s_crop = s.isig[low:high]

    # interpolar ZLP al SI recortado
    zlp_aligned = zlp.interpolate_on_axis(s_crop.axes_manager[-1])
    nav_shape = s_crop.axes_manager.navigation_shape
    zlp_broadcast = np.tile(zlp_aligned, (nav_shape[0], nav_shape[1], 1))

    zlp_fixed = s_crop.deepcopy()
    zlp_fixed.data = zlp_broadcast

    return s_crop, zlp_fixed

zlp = hs.load("ZLP_9e-5s.dm4")


s = hs.load("Aligned.hspy")
s_crop, zlp_aligned = match_axis(s, zlp)
dielectric_function(s_crop, zlp_aligned, '4-mask/otsu.txt')


