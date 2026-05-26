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
from hyperspy.axes import AxesManager


def match_axis(s, zlp, E_lims = None):
    '''
    ADVICED TO CROP ENERGY BEFOREHAND TO AVOID MEMORY PROBLEMS. 
    better to just modify zlp, as it is usually smaller than the SI
    '''
    s_crop = s.deepcopy()
    zlp_crop = zlp.deepcopy()
    
    # most restrictive energy range (overlap) between SI and ZLP
    if E_lims is not None:
        s_crop = s_crop.isig[E_lims[0]:E_lims[1]]
        zlp_crop = zlp_crop.isig[E_lims[0]:E_lims[1]]
        high = E_lims[1]
        low = E_lims[0]

    else:
        s_axis = s.axes_manager[-1].axis
        zlp_axis = zlp.axes_manager[-1].axis
        low = max(s_axis.min(), zlp_axis.min())
        high = min(s_axis.max(), zlp_axis.max())
        s_crop = s_crop.isig[low:high]
        zlp_crop = zlp_crop.isig[low:high]
    # print("After cropping:")
    # print("SI range:", s_crop.axes_manager[-1].axis.min(), "to", s_crop.axes_manager[-1].axis.max())
    # print("ZLP range:", zlp_crop.axes_manager[-1].axis.min(), "to", zlp_crop.axes_manager[-1].axis.max())
    # print("SI shape:", s_crop)
    # print("ZLP shape:", zlp_crop)

    # print("Energy scale used for interpolation:", e_scale)
    good_axis = s_crop.axes_manager.signal_axes[0]
    zlp_crop= zlp_crop.interpolate_on_axis(
    good_axis,
    axis=-1
    )

    # print("After rebinning:")
    # print("SI range:", s_crop.axes_manager[-1].axis.min(), "to", s_crop.axes_manager[-1].axis.max())
    # print("ZLP range:", zlp_crop.axes_manager[-1].axis.min(), "to", zlp_crop.axes_manager[-1].axis.max())
    # print("SI shape:", s_crop)  
    # print("ZLP shape:", zlp_crop)

    return s_crop, zlp_crop



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
    s_copy = s.deepcopy()  # create a copy of the original spectrum to avoid modifying it directly
    # Load the binary mask from the text file
    mask = np.loadtxt(mask_file, dtype=int)
    if len(s_copy.data.shape) == 2:
        s_copy.data = s_copy.data * (mask)  # Apply the mask to the spectrum data (element-wise multiplication)
    else:
        for k in range(s_copy.data.shape[-1]):
            s_copy.data[...,k] = s_copy.data[...,k] * (mask)  # Apply the mask to each spectrum in the stack
    # s_copy.plot()
    # plt.show()
    return s_copy

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
def dielectric_function(s, zlp, dir_mask, thickness, dir_output):

    s_crop, zlp_new = match_axis(s, zlp)
    
    zlp_int = zlp_new.integrate1D(axis=0)
    s_crop_masked = apply_mask(s_crop, dir_mask)
    thickness_masked = apply_mask(thickness, dir_mask)
    t = np.mean(thickness_masked.data[thickness_masked.data > 0])  # calculate the mean thickness from the masked thickness map, excluding zero values
    

    diel = s_crop_masked.kramers_kronig_analysis(zlp=zlp_int.data[0], iterations=1, n=None, t=t, delta=0.5, full_output=False)
    diel.save(f"{dir_output}dielectric_function.hspy", overwrite=True)
    diel.plot()
    plt.show()
    
    return




if __name__ == "__main__":
    zlp = hs.load("ZLP_9e-5s_1000frames.dm4")
    s = hs.load("stem.dm4")[-1]
    
    

# t = s.isig[0].copy() 

# # 3. Inyectar los datos del espesor en ese clon
# # Al ser s.isig[0], ya tiene dimensiones (422, 321) y los ejes perfectos
# t.data = thickness

# # Limpieza de seguridad
# t.data[t.data <= 0] = 1e-9

# # Cambiar el nombre para que no se confunda con EELS
# t.metadata.General.title = "Thickness Map"
# dielectric_function(s_crop, zlp_aligned, '4-mask/otsu.txt', thickness)


