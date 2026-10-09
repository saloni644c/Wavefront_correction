import numpy as np
from scipy.optimize import fsolve
from scipy.special import j1
import matplotlib.pyplot as plt

# Function to find solutions of J_1(x) = 0
def find_roots_j1_zero(initial_guesses):
    """
    Finds the roots of the equation J_1(x) = 0 using initial guesses.

    Parameters:
        initial_guesses (list of float): List of initial guesses for the roots.

    Returns:
        roots (list of float): List of roots found.
    """
    def equation(x):
        return j1(x)
    
    roots = [fsolve(equation, guess)[0] for guess in initial_guesses]
    
    print("Roots of the equation J_1(x) = 0 are:")
    for root in roots:
        print(f"x = {root:.10f}")
    
    return roots

# Function to find the root of the equation J_1(x) = x / (2 * sqrt(2))
def find_root_j1_rhs(initial_guess):
    """
    Finds the root of the equation J_1(x) = x / (2 * sqrt(2)) using an initial guess.

    Parameters:
        initial_guess (float): Initial guess for the root.

    Returns:
        root (float): Root found.
    """
    def equation(x):
        return j1(x) - (x / (2 * np.sqrt(2)))
    
    root = fsolve(equation, initial_guess)
    
    print(f"Root of the equation J_1(x) = x / (2*sqrt(2)) is x = {root[0]:.10f}")
    
    return root[0]

def get_solution_info():
    """
    Demonstrates finding the roots of the equations J_1(x) = 0 and J_1(x) = x / (2 * sqrt(2)).
    
    Example Output:
    Roots of the equation J_1(x) = 0 are:
    x = 0.0000000000
    x = 3.8317059702
    x = 7.0155866698
    x = 10.1734681351
    Root of the equation J_1(x) = x / (2*sqrt(2)) is x = 1.6163399483
    """
    # Example usage
    initial_guesses = [0, 3, 7, 11]  # Initial guesses can be adjusted as needed
    roots_j1_zero = find_roots_j1_zero(initial_guesses)

    initial_guess_rhs = 1.0  # Initial guess can be adjusted as needed
    root_j1_rhs = find_root_j1_rhs(initial_guess_rhs)

# Define the Airy disk intensity distribution function
def airy_disk_intensity(r, I0, k, h0):
    """
    Calculates the Airy disk intensity at a given radius r.

    Parameters:
    - r (array-like or float): Radius array or single value.
    - I0 (float): Maximum intensity at the center of the disk.
    - k (float): Wavenumber, defined as 2π / λ where λ is the wavelength of light.
    - h0 (float): Background intensity.

    Returns:
    - Intensity (array-like or float): Intensity at the given radius.
    """
    # Calculate the intensity distribution using the Airy function
    return I0 * (2 * j1(k * r) / (k * r))**2 + h0


def gauss_disk_intensity(R, I0, sigma, h0):
    """
    Calculates the Gaussian intensity at a given relative radius R.

    Parameters:
    - R (array-like or float): Relative radius array or single value.
    - I0 (float): Maximum intensity at the center of the disk.
    - sigma (float): Standard deviation for the Gaussian distribution.
    - h0 (float): Background intensity.

    Returns:
    - Intensity (array-like or float): Intensity at the given relative radius.
    """
    # Calculate the intensity distribution using the Gaussian function
    return I0 * np.exp(- (R**2) / (2 * sigma**2)) + h0

def get_airy_img(img, I0, k, h0, ideal_x, ideal_y, mode = 'airy',plot=True, assist=True, plot_config=None, threshold = None):
    """
    Generates and optionally plots the ideal Airy disk intensity distribution.

    Parameters:
    - img (array-like): Input image data.
    - I0 (float): Maximum intensity at the center of the disk.
    - k (float): Wavenumber, defined as 2π / λ where λ is the wavelength of light.
      k (float): sigma for gauss distribution
    - h0 (float): Background intensity.
    - ideal_x (int): X-coordinate of the ideal center.
    - ideal_y (int): Y-coordinate of the ideal center.
    - mode: use which function for optiomazation
         - 'airy' : airy function
         - 'gauss' : gauss function 
    - plot (bool): Whether to plot the image.
    - assist (bool): Whether to assist with additional annotations on the plot.
    - plot_config (tuple): Configuration for plot annotations (mean_x, mean_y, FWHM_radius, Airy_radius).
    - threshold(float): use threshold to filter

    Returns:
    - ideal_img (array-like): Generated ideal Airy disk intensity distribution.
    """
    # Calculate mean X, Y index coordinates using probability density
    indices = np.indices(img.shape)
    R = np.sqrt(( indices[1] - ideal_x )**2 + ( indices[0] - ideal_y )**2)

    #Calculate the intensity distribution of the Airy disk for a given radius
    if mode == 'airy':
        ideal_img = airy_disk_intensity(R, I0, k, h0)
    if mode == 'gauss':
        ideal_img = gauss_disk_intensity(R, I0, k, h0)

    if plot:
        # Plot the image and the circle
        fig, ax = plt.subplots()
        if not assist: 
            if threshold == None:
                img_plot = ax.imshow(ideal_img, cmap='viridis', interpolation='none')
            else:
                img_plot = ax.imshow(ideal_img > threshold, cmap='viridis', interpolation='none')
        else:
            if threshold == None:
                img_plot = ax.imshow(ideal_img + img, cmap='viridis', interpolation='none')
            else:
                img_plot = ax.imshow((ideal_img>threshold[1]) + (img>threshold[0]), cmap='viridis', interpolation='none')
        if assist: 
            mean_x,mean_y,FWHM_radius,Airy_radius = plot_config
            ax.scatter(mean_x, mean_y, c='r', s=20)  # Plot the center of the circle
            ax.scatter(ideal_x, ideal_y, c='b', s=20)  # Plot the center of target
            circle1 = plt.Circle((mean_x, mean_y), FWHM_radius, color='r', fill=False)
            ax.add_patch(circle1)
            circle2 = plt.Circle((mean_x, mean_y), Airy_radius, color='r', fill=False)
            ax.add_patch(circle2)
        
        # Set the x and y axis limits to center +/- 200
        ax.set_xlim(ideal_x - 100, ideal_x + 100)
        ax.set_ylim(ideal_y - 100, ideal_y + 100)
        
        plt.colorbar(img_plot)  # Add colorbar based on the image plot
        plt.title("2D ideal Matrix Visualization")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.savefig('./advanced_slm/figure/ideal_light_figure.jpg')
        plt.show()
    
    return ideal_img