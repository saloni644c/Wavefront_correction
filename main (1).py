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
        error = calculate_area(img, threshold=maxinum/2)
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
    distance_mode = 'Area'
    minus = True
    sleep_time = 0.01  # Time to sleep between operations

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
    Loss_config = [60, 210]
    distance_mode = 'Dice'
    minus = True
    sleep_time = 0.01  # Time to sleep between operations
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
            init_error = Error_function(img_init, ideal_img, mode=distance_mode, minus=minus, config=Loss_config)
            if result_option:
                print(f"       Loss function = {init_error} k_para = {k}")
            # Find the gradient diraction
            # for Znm = Z(1,-1) index 1
            # positive direction
            Zernike.Zernike_controller(commander,command='add',input= step_length,config=1,para='c')
            time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
            img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
            positive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
                plot=False, assist=False, plot_config=plot_config)
            positive_error = Error_function(img_init, positive_img, mode=distance_mode, minus = minus, config=Loss_config )
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
                img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                negitive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
                    plot=False, assist=False, plot_config=plot_config)
                negitive_error = Error_function(img_init, negitive_img, mode=distance_mode, minus = minus, config=Loss_config )
                if negitive_error < init_error:
                    direction_1 = -1
                    init_error = negitive_error
                    Z_1 = Zernike.Zernike_controller(commander,command='return',input=None,config=1,para='c')
                    if print_option:
                        print(f">>> to Loss function = {init_error} Z_{1} {Z_1 + step_length} -(-)-> {Z_1}")
                else:
                    Zernike.Zernike_controller(commander,command='add',input=+step_length,config=1,para='c')
                    time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                    img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                    direction_1 = 0
            # for Znm = Z(1,1) index 2
            # positive direction
            Zernike.Zernike_controller(commander,command='add',input= step_length,config=2,para='c')
            time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
            img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
            positive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
                plot=False, assist=False, plot_config=plot_config)
            positive_error = Error_function(img_init, positive_img, mode=distance_mode, minus = minus, config=Loss_config )
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
                img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
                negitive_img = Airy.get_airy_img(img_init, I0, k, h0, ideal_x, ideal_y, mode = 'gauss',
                    plot=False, assist=False, plot_config=plot_config)
                negitive_error = Error_function(img_init, negitive_img, mode=distance_mode, minus = minus, config=Loss_config )
                if negitive_error < init_error:
                    direction_2 = -1
                    init_error = negitive_error
                    Z_2 = Zernike.Zernike_controller(commander,command='return',input=None,config=2,para='c')
                    if print_option:
                        print(f">>> to Loss function = {init_error} Z_{2} {Z_2 + step_length} -(-)-> {Z_2}")
                else:
                    Zernike.Zernike_controller(commander,command='add',input=+step_length,config=2,para='c')
                    time.sleep(sleep_time)  # Sleep for a while to avoid busy-waiting
                    img_init = Img.img_read(commander,cal=False,plot=False,print_option=False)
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

def thread_wrap(func):
    @functools.wraps(func)
    def wrapper_fun(*args, **kwargs):
        worker = threading.Thread(target=func, args = args, kwargs = kwargs)
        worker.start()
    return (wrapper_fun)

def optimizer():
    pass

def test(commander):
    img , plot_config = Img.img_read(commander)
    mean_x,mean_y,FWHM_radius,Airy_radius = plot_config
    Airy.get_solution_info()
    Zernike.Zernike_controller(commander,command='print',para='c')
    I0 = 150
    k = 1.6163399483/FWHM_radius
    h0 = 0
    ideal_x = 512
    ideal_y = 512
    ideal_img = Airy.get_airy_img(img,I0,k,h0,ideal_x,ideal_y,plot_config=plot_config)


# optimization of peak intensity and width 

#a control fucntion to control zernikes
def control(n,x,p):
    if p:
        return(c.hologram.zernikes[n].c)
    else:
        c.hologram.zernikes[n].c=x

#defining the focus position for the peak intensity
def focus_intensity(step, e):
    global peak0, peak11, peak22

    peak0 = mmax()  # Initial peak measurement
    focus_center = control(4, 0, 1)

    for _ in range(5):
        focus1 = focus_center + step
        focus2 = focus_center - step

        # Test focus1
        control(4, focus1, 0)
        time.sleep(1/7)
        peak1 = mmax()

        # Test focus2
        control(4, focus2, 0)
        time.sleep(1/7)
        peak2 = mmax()

        if peak0[1] > peak1[1] and peak0[1] > peak2[1]:
            # Peak intensity is better at focus_center +/- step
            control(4, focus_center + step, 0)
            if e == 1:
                peak11 = peak0
            elif e == 2:
                peak22 = peak0
            return
        
        # Update focus_center and peak0 based on comparisons
        if peak1[1] > peak2[1]:
            focus_center, peak0 = focus1, peak1
        else:
            focus_center, peak0 = focus2, peak2
        
        if e == 1:
            peak11 = peak0
        elif e == 2:
            peak22 = peak0

    return

# Adjust peak intensity by optimizing Zernike coefficients
def adjust_peak_intensity(step, n, max_iter=10):
    """
    Adjusts the peak intensity by optimizing Zernike coefficients.

    Parameters:
        step (float): The step size for adjustment.
        n (int): The index for Zernike coefficient adjustment.
        max_iter (int): Maximum iterations for optimization loop.

    Returns:
        tuple: Status message and updated peak values.
    """
    focus_intensity(step, 0)
    
    peak = peak0

    for _ in range(max_iter):
        time.sleep(1/7)
        print("Peak intensity adjustment:", peak)
        
        fo = control(4, 0, 1)
        zernike1 = control(n, 0, 1) + step
        zernike2 = control(n, 0, 1) - step
        #test zernike1
        control(n, zernike1, 0)
        time.sleep(1/7)
        focus(step, 1)
        time.sleep(1/7)
        fo1 = control(4, 0, 1)
        control(4, fo, 0)
        #test zernike2
        control(n, zernike2, 0)
        time.sleep(1/7)
        focus(step, 2)

        #compare peaks
        peak11 = [0, 0]  # Placeholder value for peak after step 1
        peak22 = [0, 0]  # Placeholder value for peak after step 2
        
        if peak[1] < peak11[1] and peak[1] < peak22[1]:
            control(n, zernike2 + step, 0)
            control(4, fo, 0)
            return 'nothing changed', peak
        elif peak11[1] < peak22[1]:
            control(n, zernike1, 0)
            control(4, fo1, 0)
            peak = peak11
        else:
            peak = peak22

    return None, peak

# Adjust width by optimizing Zernike coefficients
# first defining focus for it, followed by adjusted width function

def adjust_width(step, max_iter=10):
    """
    Adjusts the width by optimizing Zernike coefficients.

    Parameters:
        step (float): The step size for adjustment.
        max_iter (int): Maximum iterations for optimization loop.

    Returns:
        tuple: Status message and updated width values.
    """
    focus2(step, 0)
    width = width0

    for _ in range(max_iter):
        print("Width adjustment:", width)
        
        fo = control(4, 0, 1)
        zernike1 = control(12, 0, 1) + step
        zernike2 = control(12, 0, 1) - step
        control(12, zernike1, 0)
        time.sleep(0.1)
        focus2(step, 1)
        fo1 = control(4, 0, 1)
        control(4, fo, 0)
        control(12, zernike2, 0)
        time.sleep(0.1)
        focus2(step, 2)

        width11 = 0  # Placeholder value for width after step 1
        width22 = 0  # Placeholder value for width after step 2
        
        if width < width11 and width < width22:
            control(12, zernike2 + step, 0)
            control(4, fo, 0)
            return 'nothing changed', width
        elif width11 < width22:
            control(12, zernike1, 0)
            control(4, fo1, 0)
            width = width11
        else:
            width = width22

    return None, width

def your_function():
    # Acquire the commander instance within a lock for thread safety
    with commander_lock:
        local_commander = commander
    
    # Ensure the commander is initialized
    if local_commander is None:
        time.sleep(1)  # Wait for the commander to be initialized
        return
    