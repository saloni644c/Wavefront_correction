import numpy as np
import matplotlib.pyplot as plt


def img_plot(img, plotname, limitation=False, x_limit=None, y_limit=None):
    """
    Plots a 2D image with optional axis limitations.

    Parameters:
    - img (array-like): Image data to be plotted.
    - plotname (str): Name for the plot title and saved file.
    - limitation (bool): Whether to limit the x and y axes.
    - x_limit (tuple of int): Limits for the x-axis (min, max).
    - y_limit (tuple of int): Limits for the y-axis (min, max).
    """
    # Plot the image
    fig, ax = plt.subplots()
    img_plot = ax.imshow(img, cmap='viridis', interpolation='none')
        
    if limitation:
        # Set the x and y axis limits
        ax.set_xlim(x_limit[0], x_limit[1])
        ax.set_ylim(y_limit[0], y_limit[1])
    
    plt.colorbar(img_plot)  # Add colorbar based on the image plot
    plt.title(f"2D plot of {plotname}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(f'./advanced_slm/figure/{plotname}.jpg')
    plt.show()

def img_save(img, name, path):
    np.save("./advanced_slm/measure/"+path+f"/{name}.npy",img)
    return

def img_read(commander, cal=True, plot=True,print_option=False,threshold = None):  
    """
    Reads and processes an image from the commander object, calculates key statistics,
    and optionally plots the image with additional annotations.

    Parameters:
    - commander (object): Object containing the image data.
    - cal (bool): Whether to perform calculations on the image.
    - plot (bool): Whether to plot the image.
    - threshold(float): use threshold to filter

    Returns:
    - img (array-like): Image data as a NumPy array.
    - plot_config (list): Configuration for plotting annotations if cal is True.
    """
    img = commander.image_bfp.data_roi
    img = np.array(img)
    if cal:
        if print_option:
            print(f"Image shape = {list(img.shape)}")
        
        # Get maximum pixel intensity
        max_intensity = img.max()
        if print_option:
            print(f"Max intensity = {max_intensity}")
        
        # Get half maximum pixel intensity
        half_max_intensity = max_intensity / 2
        if print_option:
            print(f"Half max intensity = {half_max_intensity}")
        
        # Filtered img by half maximum intensity
        filtered_img = (img > half_max_intensity)
        
        # Calculate probability density
        prob_density = filtered_img / filtered_img.sum()
        
        # Calculate mean X, Y index coordinates using probability density
        indices = np.indices(filtered_img.shape)
        mean_x = np.sum(indices[1] * prob_density)
        mean_y = np.sum(indices[0] * prob_density)
        if print_option:
            print(f"mean position x = {mean_x} , y = {mean_y}")
        
        # Calculate the radius using the area method
        area = filtered_img.sum()
        FWHM_radius = np.sqrt(area / np.pi)
        Airy_radius = FWHM_radius * (3.8317059702 / 1.6163399483)
        if print_option:
            print(f"Radius FWHM = {FWHM_radius} , Airy = {Airy_radius}")
        
        plot_config = [mean_x, mean_y, FWHM_radius, Airy_radius]

    if plot:
        # Plot the image and the circle
        fig, ax = plt.subplots()
        if threshold == None:
            img_plot = ax.imshow(img, cmap='viridis', interpolation='none')
        elif threshold == 'half':
            threshold = img.max()/2
            img_plot = ax.imshow(img>threshold, cmap='viridis', interpolation='none')
        else:
            img_plot = ax.imshow(img>threshold, cmap='viridis', interpolation='none')
        if cal:
            ax.scatter(mean_x, mean_y, c='r', s=20)  # Plot the center of the circle
            circle1 = plt.Circle((mean_x, mean_y), FWHM_radius, color='r', fill=False)
            ax.add_patch(circle1)
            circle2 = plt.Circle((mean_x, mean_y), Airy_radius, color='r', fill=False)
            ax.add_patch(circle2)
            
            # Set the x and y axis limits to center +/- 200
            ax.set_xlim(mean_x - 100, mean_x + 100)
            ax.set_ylim(mean_y - 100, mean_y + 100)
        
        plt.colorbar(img_plot)  # Add colorbar based on the image plot
        plt.title("2D Matrix Visualization with Circle")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.savefig('./advanced_slm/figure/light_figure.jpg')
        plt.show()

    if cal:
        return img, plot_config
    else:
        return img

def imaging_radius(img,print_option=False,propotion=0.95,
        path="./advanced_slm/result.txt"):
    threshold_value = img.max() / 10
    mask = img >= threshold_value
            
    # Calculate probability density
    prob_density = mask /mask.sum()
    
    # Calculate mean X, Y index coordinates using probability density
    indices = np.indices(mask.shape)
    mean_x = np.sum(indices[1] * prob_density)
    mean_y = np.sum(indices[0] * prob_density)
    R_relative = (mask * np.sqrt((indices[1]-mean_x)**2+(indices[0]-mean_y)**2)).reshape(-1)
    R_relative = R_relative[R_relative>0]
    mid_index = int(propotion*len(R_relative))
    min_radius = np.sort(R_relative)[mid_index]

    # 3. 找到圆心 (即距离变换中最大值的位置)
    center = np.array([mean_y,mean_x])
    if print_option:
        print(f"mean position x = {mean_x} , y = {mean_y}")
        print(f"Radius = {min_radius}")
        with open(path, "a", encoding="utf-8") as file:
            # 写入文字内容
            file.write(f"{min_radius}," + "\n")
    # 4. 绘制结果
    plt.figure(figsize=(8, 8))

    # 绘制掩码图像

    plt.imshow(mask, cmap='viridis', interpolation='none')
    plt.colorbar()  # Add colorbar based on the image plot

    # 绘制圆
    circle = plt.Circle(center[::-1], min_radius, color='red', fill=False, linewidth=2)
    plt.gca().add_patch(circle)

    # 标注圆心
    plt.plot(center[1], center[0], 'ro')

    plt.title(f"Center: {center}, Radius: {min_radius:.2f}")
    plt.show()



    """    plt.figure(figsize=(8, 8))    # 绘制掩码图像
    from scipy.ndimage import convolve
    laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_img = convolve(img, laplace_operator)
    diff_h = np.abs(np.diff(img, axis=1))
    diff_v = np.abs(np.diff(img, axis=0))
    mask = laplacian_img**2
    plt.imshow(mask, cmap='viridis', interpolation='none')
    plt.colorbar()  # Add colorbar based on the image plot

    # 绘制圆
    circle = plt.Circle(center[::-1], min_radius, color='red', fill=False, linewidth=2)
    plt.gca().add_patch(circle)

    # 标注圆心
    plt.plot(center[1], center[0], 'ro')

    plt.title(f"Center: {center}, Radius: {min_radius:.2f}")
    plt.show()

    print(img)"""