import advanced_slm.Airy as Airy
import advanced_slm.Img as Img
import advanced_slm.Zernike as Zernike

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import scipy.special as sc
from scipy.optimize import curve_fit
import math
import functools
import threading
from scipy.ndimage import convolve

import importlib
importlib.reload(Airy)
importlib.reload(Img)
importlib.reload(Zernike)

# Assume commander is a global variable updated by another thread
commander_lock = threading.Lock()
commander = None



def update_commander(new_commander):
    global commander
    with commander_lock:
        commander = new_commander

import numpy as np
from scipy.stats import entropy

def Error_function(img1, img2, mode='MSE', config=None, minus = False):
    """
    Calculate the error between two 2D image matrices based on the specified mode.

    Parameters:
    - img1 (ndarray): First image matrix.
    - img2 (ndarray): Second image matrix.
    - mode (str): Mode of error calculation. Options are 'MSE', 'MAE', 'PSNR', 'SSIM', 'Dice', 'KL', 'Wasserstein'.
    - config (dict): Additional configuration parameters (if any).

    Returns:
    - error (float): Calculated error between the two images.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    if mode == 'MSE':
        # Mean Squared Error
        error = np.mean((img1 - img2) ** 2)
    elif mode == 'MAE':
        # Mean Absolute Error
        error = np.mean(np.abs(img1 - img2))
    elif mode == 'Dice Times':
        # Times
        img1_flat = img1.flatten() > config
        img2_flat = img2.flatten() > config
        intersection = np.sum(img1_flat * img2_flat)
        error = 1 - (2. * intersection) / (np.sum(img1_flat) + np.sum(img2_flat))
        error += np.mean(img1*img2)/((np.max(img1))*(np.max(img2))) 
    elif mode == 'PSNR':
        from skimage.metrics import peak_signal_noise_ratio as psnr
        """
        The smaller the MSE, the larger the PSNR; therefore, the larger the PSNR, the better the image quality.
        """
        # Peak Signal-to-Noise Ratio
        if config and 'data_range' in config:
            data_range = config['data_range']
        else:
            data_range = img1.max() - img1.min()
        error = psnr(img1, img2, data_range=data_range)
    elif mode == 'SSIM':
        from skimage.metrics import structural_similarity as ssim
        # Structural Similarity Index
        """
        SSIM is symmetric, that is, SSIM(x,y)=SSIM(y,x)
        SSIM is a number between 0 and 1. The larger the value, 
            the smaller the difference between the output image and 
            the undistorted image, that is, the better the image quality. 
            When the two images are exactly the same, SSIM=1;
        """
        if config and 'data_range' in config:
            data_range = config['data_range']
        else:
            data_range = img1.max() - img1.min()
        error, _ = ssim(img1, img2, full=True, data_range=data_range)
    elif mode == 'Dice':
        # Dice Loss
        """
        Dice coefficient is used to gauge the similarity of two samples.
        """
        img1_flat = img1.flatten() > config[0]
        img2_flat = img2.flatten() > config[1]
        intersection = np.sum(img1_flat * img2_flat)
        error = 1 - (2. * intersection) / (np.sum(img1_flat) + np.sum(img2_flat))
    elif mode == 'KL':
        # KL Divergence
        """
        KL Divergence measures how one probability distribution diverges from a second expected probability distribution.
        """
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        # Normalize to create probability distributions
        img1_prob = img1_flat / np.sum(img1_flat)
        img2_prob = img2_flat / np.sum(img2_flat)
        error = entropy(img1_prob, img2_prob)
    elif mode == 'Wasserstein':
        # Wasserstein Distance (Earth Mover's Distance)
        """
        Wasserstein Distance measures the distance between two probability distributions over a region D.
        """
        img1_cdf = np.cumsum(img1.flatten())
        img2_cdf = np.cumsum(img2.flatten())
        error = np.sum(np.abs(img1_cdf - img2_cdf))
    else:
        raise ValueError("Unsupported mode. Choose from 'MSE', 'MAE', 'PSNR', 'SSIM', 'Dice', 'KL', 'Wasserstein'.")

    """    
    # Example usage:
    img1 = np.random.rand(256, 256)
    img2 = np.random.rand(256, 256)

    print("MSE:", Error_function(img1, img2, mode='MSE'))
    print("MAE:", Error_function(img1, img2, mode='MAE'))
    print("PSNR:", Error_function(img1, img2, mode='PSNR'))
    print("SSIM:", Error_function(img1, img2, mode='SSIM'))
    print("Dice:", Error_function(img1, img2, mode='Dice'))
    print("KL:", Error_function(img1, img2, mode='KL'))
    print("Wasserstein:", Error_function(img1, img2, mode='Wasserstein'))
    """
    if not minus:
        return error
    else:
        return -error

def focus_function(img,mode='Area', config=None, minus = False):
    """
    Evaluates the quality of an image based on various methods.

    Parameters:
    - img (ndarray): Input image.
    - mode (str): Method to evaluate image quality. Options are 'MSE', 'NRMSE', 'PSNR', 'SSIM', 'AREA'.
    - config (dict, optional): Additional configuration parameters.
    - minus (bool, optional): Whether to return the negative value of the calculated error/metric.

    Returns:
    - error (float): Calculated error/metric for the image.
    """
    def calculate_area(img, threshold=50):
        """
        Calculates the area of the thresholded image.

        Parameters:
        - img (ndarray): Input image.
        - threshold (float): Threshold value to filter the image.

        Returns:
        - area (float): Area of the thresholded image.
        """
        # Apply threshold
        binary_img = img > threshold
        # Calculate the area (number of pixels above the threshold)
        area = np.sum(binary_img)
        return area
    def calculate_kurtosis(data):
        from scipy.stats import kurtosis
        """
        Calculate the kurtosis of a given data distribution.

        Parameters:
        - data (array-like): Input data distribution.

        Returns:
        - kurtosis_value (float): Calculated kurtosis of the distribution.
        """
        return kurtosis(data, fisher=False).sum()
    #why need this sum() here?

    def calculate_std(data):
        """
        Calculate the standard deviation of a given data distribution.

        Parameters:
        - data (array-like): Input data distribution.

        Returns:
        - std_value (float): Calculated standard deviation of the distribution.
        """
        return np.std(data)

    def calculate_laplacian_sharpness(data):
        from scipy.ndimage import laplace
        """
        Calculate the Laplacian to evaluate the sharpness of a distribution.

        Parameters:
        - data (array-like): Input data distribution.

        Returns:
        - laplacian_value (float): Sum of the Laplacian values indicating sharpness.
        """
        laplacian_data = laplace(data)
        return np.sum(np.abs(laplacian_data))

    def calculate_local_variance(data, window_size=5):
        """
        Calculate the local variance of a given data distribution.

        Parameters:
        - data (array-like): Input data distribution.
        - window_size (int): Size of the local window to calculate variance.

        Returns:
        - local_variance (array-like): Local variance of the distribution.
        """
        local_variance = [np.var(data[max(0, i-window_size):min(len(data), i+window_size)]) for i in range(len(data))]
        return np.mean(local_variance)
    if mode == 'Area':
        # calculate area of image
        maxinum = img.max()
        error = calculate_area(img, threshold=maxinum/2)
    elif mode == 'Area_plus':
        # calculate area of image
        maxinum = img.max()
        Area = calculate_area(img, threshold=maxinum/2)/(1000)
        light_Area = calculate_area(img, threshold=maxinum/5)/(1000)
        Radius = np.sqrt(Area/np.pi)
        error = (light_Area) * (Area**2) * ( 1 + 10/maxinum)
    elif mode == 'Kurtosis':
        error = calculate_kurtosis(img)
    elif mode == 'STD':
        error = calculate_std(img)
    elif mode == 'Laplacian':
        error = calculate_laplacian_sharpness(img)
    elif mode == 'LocalVariance':
        error = calculate_local_variance(img)
    elif mode == 'MSE':
        from skimage.metrics import mean_squared_error
        error = mean_squared_error(img, config['reference']) if config and 'reference' in config else None
    elif mode == 'PSNR':
        from skimage.metrics import peak_signal_noise_ratio
        data_range = config['data_range'] if config and 'data_range' in config else img.max() - img.min()
        error = peak_signal_noise_ratio(img, config['reference'], data_range=data_range) if config and 'reference' in config else None
    elif mode == 'SSIM':
        from skimage.metrics import structural_similarity
        data_range = config['data_range'] if config and 'data_range' in config else img.max() - img.min()
        error, _ = structural_similarity(img, config['reference'], data_range=data_range, full=True) if config and 'reference' in config else None
    else:
        raise ValueError("Unsupported mode.")

    return -error if minus else error

def position_function(x1,y1,x2,y2):
    return np.sqrt(( x1 - x2 )**2+( y1 - y2 )**2)

def focus_optimizer(stage = 5, decay_rate = 1.8 ,step_length = 0.1, max_step=100):
    """
    Optimizes the focus of an image using Zernike polynomials adjustments.

    Parameters:
    - stage (int): Number of stages for the optimization process.
    - decay_rate (float): Rate at which the step length decays after each stage.
    - step_length (float): Initial step length for adjusting Zernike polynomials.
    - max_step (int): Maximum number of steps per stage.

    The function adjusts Zernike polynomials' coefficients to minimize a loss function.
    """

    # Acquire the commander instance within a lock for thread safety
    with commander_lock:
        local_commander = commander
    
    # Ensure the commander is initialized
    if local_commander is None:
        time.sleep(1)  # Wait for the commander to be initialized
        return

    # Options for debugging and output
    print_option = False
    plot_option = True
    result_option = True

    # Initialize directions for Zernike adjustments
    direction_0 = 0  # Direction for Z(0, 0) adjustment
    direction_4 = 0  # Direction for Z(2, 0) adjustment

    # Configuration for the loss function
    Loss_config = 50
    distance_mode = 'Area_plus'
    minus = False
    sleep_time = 0.5  # Time to sleep between operations

    # Optimization loop over the number of stages
    for i in range(stage):
        # Loop for each step within the stage
        for j in range(max_step):
            if result_option:
                print(f">>>>> Optimizing stage/sub_stage/step_length = {i}/{j}/{step_length}")
            
            # Read the initial image and calculate the initial error
            img_init, plot_config = Img.img_read(commander, plot=plot_option, print_option=False, threshold = 'half')
            init_error = focus_function(img_init, mode=distance_mode, config=None, minus=minus)
            
            if result_option:
                print(f"       Loss function = {init_error}")

            # Find the gradient diraction
            # for Znm = Z(0,0) index 0
            # positive direction
            Zernike.Zernike_controller(commander,command='add',input= step_length,config=0,para='c')
            time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
            img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
            positive_error = focus_function(img_init,mode=distance_mode, config=None, minus = minus )
            if positive_error < init_error:
                direction_0 = +1
                init_error = positive_error
                Z_0 = Zernike.Zernike_controller(commander,command='return',input=None,config=0,para='c')
                if print_option:
                    print(f">>> to Loss function = {init_error} Z_{0} {Z_0 - step_length} -(+)-> {Z_0}")
            else:
            # negitive direction
                Zernike.Zernike_controller(commander,command='add',input=-2*step_length,config=0,para='c')
                time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                negitive_error = focus_function(img_init,mode=distance_mode, config=None, minus = minus )
                if negitive_error < init_error:
                    direction_0 = -1
                    init_error = negitive_error
                    Z_0 = Zernike.Zernike_controller(commander,command='return',input=None,config=0,para='c')
                    if print_option:
                        print(f">>> to Loss function = {init_error} Z_{0} {Z_0 + step_length} -(-)-> {Z_0}")
                else:
                    Zernike.Zernike_controller(commander,command='add',input=+step_length,config=0,para='c')
                    time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                    img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                    direction_0 = 0
            # for Znm = Z(2,0) index 4
            # positive direction
            Zernike.Zernike_controller(commander,command='add',input= step_length,config=4,para='c')
            time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
            img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
            positive_error = focus_function(img_init,mode=distance_mode, config=None, minus = minus )
            if positive_error < init_error:
                direction_4 = +1
                init_error = positive_error
                Z_4 = Zernike.Zernike_controller(commander,command='return',input=None,config=4,para='c')
                if print_option:
                    print(f">>> to Loss function = {init_error} Z_{4} {Z_4 - step_length} -(+)-> {Z_4}")
            else:
            # negitive direction
                Zernike.Zernike_controller(commander,command='add',input=-2*step_length,config=4,para='c')
                time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                negitive_error = focus_function(img_init,mode=distance_mode, config=None, minus = minus )
                if negitive_error < init_error:
                    direction_4 = -1
                    init_error = negitive_error
                    Z_4 = Zernike.Zernike_controller(commander,command='return',input=None,config=4,para='c')
                    if print_option:
                        print(f">>> to Loss function = {init_error} Z_{4} {Z_4 + step_length} -(-)-> {Z_4}")
                else:
                    Zernike.Zernike_controller(commander,command='add',input=+step_length,config=4,para='c')
                    time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                    img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                    direction_4 = 0
                    if print_option:
                        print(f"<<<<<<< init={init_error},pos={positive_error},neg={negitive_error}")
            # If no improvement in both directions, break the inner loop
            if direction_0 == 0 and direction_4 == 0:
                break

        # Decay the step length for the next stage
        step_length = step_length / decay_rate
        time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
    
    print(">>>> optimize completed <<<<")
    # Read the initial image and calculate mean positions and radii
    img_init = Img.img_read(commander,cal=True,plot=True,print_option=False)
    return


def position_optimizer(stage = 5, decay_rate = 1.8 ,step_length = 0.1, max_step=100):
    """
    Optimizes the position of an image using Zernike polynomials adjustments.

    Parameters:
    - stage (int): Number of stages for the optimization process.
    - decay_rate (float): Rate at which the step length decays after each stage.
    - step_length (float): Initial step length for adjusting Zernike polynomials.
    - max_step (int): Maximum number of steps per stage.

    The function adjusts Zernike polynomials' coefficients to minimize a loss function.
    """

    # Acquire the commander instance within a lock for thread safety
    with commander_lock:
        local_commander = commander
    
    # Ensure the commander is initialized
    if local_commander is None:
        time.sleep(1)  # Wait for the commander to be initialized
        return

    # Options for debugging and output
    print_option = False
    plot_option = True
    result_option = False
    k_decay = 1

    # Initialize directions for Zernike adjustments
    direction_1 = 0  # Direction for Z(1, -1) adjustment
    direction_2 = 0  # Direction for Z(1, 1) adjustment

    # Configuration for the loss function
    Loss_config = [50, 200]
    distance_mode = 'Dice'
    minus = True
    sleep_time = 0.5  # Time to sleep between operations
    I0 = 250
    k = 100
    h0 = 0
    ideal_x = 512
    ideal_y = 512

    # Optimization loop over the number of stages
    for i in range(stage):
        # Loop for each step within the stage
        for j in range(max_step):
            if result_option:
                print(f">>>>> Optimizing stage/sub_stage/step_length = {i}/{j}/{step_length}")
            
            # Read the initial image and calculate mean positions and radii
            img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
            mean_x, mean_y, FWHM_radius, Airy_radius = plot_config
            
            # Calculate the new k value based on mean positions
            k = max(np.sqrt((mean_x - ideal_x)**2 + (mean_y - ideal_y)**2), FWHM_radius) / k_decay
            
            # Generate the ideal image
            ideal_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode='gauss',
                                          plot=((j==0) and plot_option), assist=True, plot_config=plot_config,threshold=Loss_config)
            
            # Calculate the initial error
            init_error = position_function(plot_config[0],plot_config[1],ideal_x,ideal_y)
            if result_option:
                print(f"       Loss function = {init_error} k_para = {k}")
            # Find the gradient diraction
            # for Znm = Z(1,-1) index 1
            # positive direction
            Zernike.Zernike_controller(commander,command='add',input= step_length,config=1,para='c')
            time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
            img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
            # positive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
            #     plot=False, assist=False, plot_config=plot_config)
            positive_error = position_function(plot_config[0],plot_config[1],ideal_x,ideal_y)
            if positive_error < init_error:
                direction_1 = +1
                init_error = positive_error
                Z_1 = Zernike.Zernike_controller(commander,command='return',input=None,config=1,para='c')
                if print_option:
                    print(f">>> to Loss function = {init_error} Z_{1} {Z_1 - step_length} -(+)-> {Z_1}")
            else:
            # negitive direction
                Zernike.Zernike_controller(commander,command='add',input=-2*step_length,config=1,para='c')
                time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
                # negitive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
                #     plot=False, assist=False, plot_config=plot_config)
                negitive_error = position_function(plot_config[0],plot_config[1],ideal_x,ideal_y)
                if negitive_error < init_error:
                    direction_1 = -1
                    init_error = negitive_error
                    Z_1 = Zernike.Zernike_controller(commander,command='return',input=None,config=1,para='c')
                    if print_option:
                        print(f">>> to Loss function = {init_error} Z_{1} {Z_1 + step_length} -(-)-> {Z_1}")
                else:
                    Zernike.Zernike_controller(commander,command='add',input=+step_length,config=1,para='c')
                    time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                    img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
                    direction_1 = 0
            # for Znm = Z(1,1) index 2
            # positive direction
            Zernike.Zernike_controller(commander,command='add',input= step_length,config=2,para='c')
            time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
            img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
            # positive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
            #     plot=False, assist=False, plot_config=plot_config)
            positive_error = position_function(plot_config[0],plot_config[1],ideal_x,ideal_y)
            if positive_error < init_error:
                direction_2 = +1
                init_error = positive_error
                Z_2 = Zernike.Zernike_controller(commander,command='return',input=None,config=2,para='c')
                if print_option:
                    print(f">>> to Loss function = {init_error} Z_{2} {Z_2 - step_length} -(+)-> {Z_2}")
            else:
            # negitive direction
                Zernike.Zernike_controller(commander,command='add',input=-2*step_length,config=2,para='c')
                time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
                # negitive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
                #     plot=False, assist=False, plot_config=plot_config)
                negitive_error = position_function(plot_config[0],plot_config[1],ideal_x,ideal_y)
                if negitive_error < init_error:
                    direction_2 = -1
                    init_error = negitive_error
                    Z_2 = Zernike.Zernike_controller(commander,command='return',input=None,config=2,para='c')
                    if print_option:
                        print(f">>> to Loss function = {init_error} Z_{2} {Z_2 + step_length} -(-)-> {Z_2}")
                else:
                    Zernike.Zernike_controller(commander,command='add',input=+step_length,config=2,para='c')
                    time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                    img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
                    direction_2 = 0
                    if print_option:
                        print(f"<<<<<<< init={init_error},pos={positive_error},neg={negitive_error}")
            # If no improvement in both directions, break the inner loop
            if direction_1 == 0 and direction_2 == 0:
                break
        # Decay the step length for the next stage
        step_length = step_length / decay_rate
        time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting

    print(">>>> optimize completed <<<<")
    # Read the initial image and calculate mean positions and radii
    img_init, plot_config = Img.img_read(commander, plot=False, print_option=False)
    mean_x, mean_y, FWHM_radius, Airy_radius = plot_config
    # Calculate the new k value based on mean positions
    k = max(np.sqrt((mean_x - ideal_x)**2 + (mean_y - ideal_y)**2), FWHM_radius) / k_decay
    # Generate the ideal image
    ideal_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode='gauss',
                                    plot=True, assist=True, plot_config=plot_config)
    return

def optim_function(img,mode='Area', config=None, minus = False):
    """
    Evaluates the quality of an image based on various methods.

    Parameters:
    - img (ndarray): Input image.
    - mode (str): Method to evaluate image quality. Options are 'MSE', 'NRMSE', 'PSNR', 'SSIM', 'AREA'.
    - config (dict, optional): Additional configuration parameters.
    - minus (bool, optional): Whether to return the negative value of the calculated error/metric.

    Returns:
    - error (float): Calculated error/metric for the image.
    """
    def calculate_area(img, threshold=50):
        """
        Calculates the area of the thresholded image.

        Parameters:
        - img (ndarray): Input image.
        - threshold (float): Threshold value to filter the image.

        Returns:
        - area (float): Area of the thresholded image.
        """
        # Apply threshold
        binary_img = img > threshold
        # Calculate the area (number of pixels above the threshold)
        area = np.sum(binary_img)
        return area
    if mode == 'Area':
        # calculate area of image
        maxinum = img.max()
        # Filtered img by half maximum intensity
        filtered_img = (img > maxinum/2)
        filtered_img2 = (img > maxinum/5)
        # Calculate probability density
        prob_density = filtered_img / filtered_img.sum()
        # Calculate mean X, Y index coordinates using probability density
        indices = np.indices(filtered_img.shape)
        mean_x = np.sum(indices[1] * prob_density)
        mean_y = np.sum(indices[0] * prob_density)
        R = np.sqrt(( indices[1] - mean_x )**2 + ( indices[0] - mean_y )**2)
        # Calculate the radius using the area method
        area = filtered_img.sum()
        FWHM_radius = np.sqrt(area / np.pi)
        Airy_radius = FWHM_radius * (3.8317059702 / 1.6163399483)
        R = ( 100*R + R**2 ) / ( 1 + np.exp(R/(3*Airy_radius)) )
        symmetry1 = np.sqrt(np.sum( (R/1000)   * (filtered_img2/100) ))
        symmetry2 = np.sqrt( 1 +   
                        np.abs(   np.sum((filtered_img2) * (indices[1]>mean_x)) - 
                            np.sum((filtered_img2) * (indices[1]<mean_x))) + 
                        np.abs(   np.sum((filtered_img2) * (indices[0]>mean_y)) - 
                            np.sum((filtered_img2) * (indices[0]<mean_y))))
        error = (symmetry2) * (symmetry1) * ( (100/maxinum)**3 )
    elif mode == 'Area_plus':
        # calculate area of image
        maxinum = img.max()
        Area = calculate_area(img, threshold=maxinum/2)
        error = (Area)/(maxinum**2)

    elif mode == 'SMD':
        # SMD 模式
        diff_h = np.abs(np.diff(img, axis=1))
        diff_v = np.abs(np.diff(img, axis=0))
        error = np.sum(diff_h) + np.sum(diff_v)

    elif mode == 'Roberts':
        # Roberts 模式
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        Gx = convolve(img, roberts_x)
        Gy = convolve(img, roberts_y)
        error = np.sum(np.sqrt(Gx**2 + Gy**2))

    elif mode == 'Brenner':
        # Brenner 模式
        diff_h = np.diff(img, n=2, axis=1)
        diff_v = np.diff(img, n=2, axis=0)
        error = np.sum(diff_h**2) + np.sum(diff_v**2)

    elif mode == 'Laplace':
        # Laplace
        img = img * (img>50)
        laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian_img = convolve(img, laplace_operator)
        N_grad = np.sum(laplacian_img**2 > 200)
        error = np.sum(laplacian_img**2/N_grad)


    else:
        raise ValueError("Unsupported mode.")

    return -error if minus else error
    
def Zernike_optimizer(stage = 5, decay_rate = 1.8 ,step_length = 0.1, 
                      max_step=100,optim_list = 'default'):
    """
    Optimizes the focus of an image using Zernike polynomials adjustments.

    Parameters:
    - stage (int): Number of stages for the optimization process.
    - decay_rate (float): Rate at which the step length decays after each stage.
    - step_length (float): Initial step length for adjusting Zernike polynomials.
    - max_step (int): Maximum number of steps per stage.

    The function adjusts Zernike polynomials' coefficients to minimize a loss function.
    """

    # Acquire the commander instance within a lock for thread safety
    with commander_lock:
        local_commander = commander
    
    # Ensure the commander is initialized
    if local_commander is None:
        time.sleep(1)  # Wait for the commander to be initialized
        return

    if optim_list == 'default':
        optim_list = [3,5,6,7,8,9,10,11,12,13,14]
    elif optim_list == 'pa':
        optim_list = [0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    elif optim_list == 'all':
        optim_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    elif optim_list == 'small_default':
        optim_list = [3,5,6,7,8,9]
    elif optim_list == 'tiny_default':
        optim_list = [3,5]
    elif optim_list == '2':
        optim_list = [3,5]
    elif optim_list == '3':
        optim_list = [6,7,8,9]
    elif optim_list == '4':
        optim_list = [10,11,12,13,14]
    elif optim_list == '5':
        optim_list = [15,16,17,18,19,20]
    elif optim_list == 'f':
        optim_list = [0,4,21,12,22]
    elif optim_list == 's':
        optim_list = [3,5,6,9,10,14,15,20]
    elif optim_list == '23':
        optim_list = [3,5,6,7,8,9]
    optim_list = np.array(optim_list,dtype=np.int32)

    # Options for debugging and output
    print_option = False
    plot_option = True
    result_option = True

    # Initialize directions for Zernike adjustments
    direction = np.array([0]*(np.max(optim_list)+1))  # Direction for Z(0, 0) adjustment

    # Configuration for the loss function
    Loss_config = 50
    distance_mode = 'Area' # 'Area'
    minus = False
    sleep_time = 0.5  # Time to sleep between operations

    # Optimization loop over the number of stages
    for i in range(stage):
        # Loop for each step within the stage
        for j in range(max_step):
            if result_option:
                print(f">>>>> Optimizing stage/sub_stage/step_length = {i}/{j}/{step_length}")
            
            # Read the initial image and calculate the initial error
            # img_init, plot_config = Img.img_read(commander, plot=(j==0 and plot_option), print_option=False, threshold = None)
            img_init = Img.img_read(commander, cal=False, plot=(j==0 and plot_option), print_option=False, threshold = None)
            init_error = optim_function(img_init, mode=distance_mode, config=None, minus=minus)
            
            if result_option:
                print(f"       Loss function = {init_error}")
            
            for Z_id in optim_list:
                # Find the gradient diraction
                # for Znm = index Z_id
                # positive direction
                Z_id = int(Z_id)
                Zernike.Zernike_controller(commander,command='add',input= step_length,config=Z_id,para='c')
                time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                positive_error = optim_function(img_init,mode=distance_mode, config=None, minus = minus )
                if positive_error < init_error:
                    Z_i = Zernike.Zernike_controller(commander,command='return',input=None,config=Z_id,para='c')
                    if print_option:
                        print(f"            >>>  Loss function = {init_error} -----> {positive_error}")
                        print(f"                 Z_{Z_id} {Z_i - step_length} -(+)-> {Z_i}")
                    direction[Z_id] = +1
                    init_error = positive_error
                else:
                # negitive direction
                    Zernike.Zernike_controller(commander,command='add',input=-2*step_length,config=Z_id,para='c')
                    time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                    img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                    negitive_error = optim_function(img_init,mode=distance_mode, config=None, minus = minus )
                    if negitive_error < init_error:
                        Z_i = Zernike.Zernike_controller(commander,command='return',input=None,config=Z_id,para='c')
                        if print_option:
                            print(f"            >>>  Loss function = {init_error} -----> {negitive_error}")
                            print(f"                 Z_{Z_id} {Z_i - step_length} -(+)-> {Z_i}")
                        direction[Z_id] = -1
                        init_error = negitive_error     
                    else:
                        Zernike.Zernike_controller(commander,command='add',input=+step_length,config=Z_id,para='c')
                        time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                        img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                        direction[Z_id] = 0
                        if print_option:
                            print(f"            <<<<<<< init={init_error},pos={positive_error},neg={negitive_error}")
            # If no improvement in both directions, break the inner loop
            if np.sum(abs(direction)) == 0:
                break

        # Decay the step length for the next stage
        step_length = step_length / decay_rate
        time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
    
    print(">>>> optimize completed <<<<")
    # Read the initial image and calculate mean positions and radii
    img_init = Img.img_read(commander,cal=True,plot=True,print_option=False)
    return

def read_info():
    # Acquire the commander instance within a lock for thread safety
    with commander_lock:
        local_commander = commander
    
    # Ensure the commander is initialized
    if local_commander is None:
        time.sleep(1)  # Wait for the commander to be initialized
        return
    
    img , plot_config = Img.img_read(commander, cal=True, plot=False,print_option=True,threshold = None)
    #img  = Img.img_read(commander, cal=False, plot=True,print_option=False,threshold = 10)
    Img.imaging_radius(img,print_option=True,path="./advanced_slm/measure/250nm/2000result.txt")
    Img.img_save(img,"2000nm_img","250nm") 
    #img  = Img.img_read(commander, cal=False, plot=True,print_option=False)
    mean_x,mean_y,FWHM_radius,Airy_radius = plot_config