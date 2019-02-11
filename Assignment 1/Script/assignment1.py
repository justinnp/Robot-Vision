import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
#use for 2d convolutions
from scipy import ndimage, signal
from PIL import Image
from statistics import median

#Author: Justin Powell
#CAP4453 - Spring 2019
#Assignment 1

# Main Function
def main():
    print("------------------------------------------------------------------------------------------")
    print("Justin Powell")
    print("Assignment 1")
    print("CAP4553 - Spring 2019")  
    print("\nDescription: Various image filters used to process images")
    print("\nInputs: Inputs are stored within the project folder")
    print("\nOutputs: Outputs will be stored within a folder named from the")
    print("specific filter used. All folders will be contained within the project folder.")
    print("------------------------------------------------------------------------------------------")  

    #box filter
    print("Applying Box Filter. Images will be saved in '/box_filters'")
    print("Description:")
    print("The box filtered images are the result of finding the weighted neighboring average and applying it")
    print("to the central pixel, over the entire image. This results in a smoothing effect that looks blurred")
    print("increasing the kernel size seems to create a greater smoothing effect, resulting in a more blurred yet sharper image")
    box_filter()
    print("\n")

    #median filter
    print("Applying Median Filter. Images will be saved in '/median_filters'")
    print("Description:")
    print("The median filter removes noise from an image. We can expect as the kernel")
    print("increases the noise will decrease proportionally since there was less noise in")
    print("the second image. The filtered second image appears to smooth.")
    print("As the kernel size increases, we spread out into more pixels.")
    median_filter()
    print("\n")

    print("Applying Gaussian Filter. Images will be saved in '/gaussian_filters'")
    print("Description:")
    print("The Gaussian filter uses a gaussian function with a determined sigma to convolve the input image.")
    print("The Gaussian filter is helpful when reducing noise, reducing detail and adding blur.")
    print("As the standard deviation, sigma, used in the gaussian fuction increases, the reduction of noise and detail also increase, as does blur.")
    gaussian_filter()
    print("\n")

    print("Applying Gradient Operations. Images will be saved in '/gradient_filters'")
    print("Description:")
    print("The gradient operations that are being used as filters, find the gradient or change in the tonal intensity")
    print("the forward gradient in both axis directions is a mirror image of the backward gradient") 
    print("the central gradient takes the hypotenuse of the forward x axis and the backward y axis, as well as forward y and backward x")
    gradient_operation()
    print("\n")

    print("Applying Sobel Filter. Images will be saved in '/sobel_filters'")
    print("Description:")
    print("Our sobel filter emphasizes the edges in our input images in our output images,")
    print("the sobel filter emphasizes the horizontal edges as well as the vertical edges additonally, the sobel filter finds")
    print("the hypotenuse of the horizontal and vertical edges then produces a complete sobel processed image.")
    sobel_filter()
    print("\n")

    print("Applying Fast Gaussian Filter. Images will be saved in '/fast_gaussian'")
    print("Description:")
    print("The Fast Gaussian is more efficient as we can apply one directional gradient to an image")
    print("then apply the next directional gradient to the already processed image.")
    fast_gaussian()
    print("\n")

    print("Creating the Histogram. Plots will be saved in '/histograms'")
    print("Description:")
    print("The Histogram shows the tonal frequency of pixels across the image. The histogram uses bins to separate ranges.")
    print("As we increase the bin size we get the full range of colors and more detail in each histogram")
    print("the histograms are more representiative of the image's true color, there are more ranges thus more accuracy.")
    print("\n")
    print("\n")
    histogram()
    print("\n")
    print("\n")

    print("Applying the Threshold. Images will be saved in '/entropy_threshold'")
    print("Description:")
    print("Using a specific threshold, we can turn a greyscale 0-255 image into a binary image.")
    print("Based on the threshold, pixels with a value above it will be set to 255 (white),")
    print("while pixels below the threshold will be set to 0 (black). Resulting in a binary image.")
    entropy_threshold()
    print("\n")

    print("Applying the Canny Edge Detection. Plots will be saved in '/canny'")
    print("Description:")
    print("The Canny Edge Detection tool uses a series of algorithms to detect in depth information in the image.")
    print("Using said series of algorithms, the Canny operator can detect a wide range of edges within the image.")
    print("Canny takes into account our gaussian filtering to smooth and blur the input image. Once the image has")
    print("been blurred, we then apply our gradient operations to the image, to ultimately process an image magnitude with a specific sigma.")
    print("Canny seems to work optimally when a median standard deviation is used, no outliers, not too low nor too high.")
    print("We then use suppression to remove anomalies occuring from edge detection. We then use a threshold to determine genuine edges and track by hysteresis.")
    canny_detection()
    print("\n")

# Create the folder where we will store our output 
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

###################################### 1. Box Filtering ###############################################
def box_filter():
    path = "box_filters/"
    create_folder(path)
    #create the kernels for the filter
    kernel_3x3 = create_box_kernel(3)
    kernel_5x5 = create_box_kernel(5)
    #image names
    image1_filename = "image1.png"
    image2_filename = "image2.png"
    #read the image as greyscale to get the corresponding 2d array
    image1 = cv2.imread(image1_filename,0)
    image2 = cv2.imread(image2_filename,0)
    #find the height and width of the images
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    #using the convolve function from ndimage in scipy, we can filter the image and smooth it
    #tried using convolve in numpy but the array needs to be 1d, so reshaping needs to happen
    #constat and cval take into account edges and padding
    #filter the images using a 3x3 kernel
    image1_box_3x3 = ndimage.convolve(image1, kernel_3x3, mode='constant', cval=0.0)
    image2_box_3x3 = ndimage.convolve(image2, kernel_3x3, mode='constant', cval=0.0)
    #write the images
    cv2.imwrite("./" + path + "image1_box_3x3.png", image1_box_3x3)
    cv2.imwrite("./" + path + "image2_box_3x3.png", image2_box_3x3)
    #filter the image using a 5x5 kernel
    image1_box_5x5 = ndimage.convolve(image1, kernel_5x5, mode='constant', cval=0.0)
    image2_box_5x5 = ndimage.convolve(image2, kernel_5x5, mode='constant', cval=0.0)
    #write the images
    cv2.imwrite("./" + path + "image1_box_5x5.png", image1_box_5x5)
    cv2.imwrite("./" + path + "image2_box_5x5.png", image2_box_5x5)
#the filtered images are the result of finding the weighted neighboring average and applying it
#to the central pixel, over the entire image. This results in a smoothing effect that looks blurred
#increasing the kernel size seems to create a greater smoothing effect, resulting in a more blurred yet sharper image

# Kernel Creation 
def create_box_kernel(x):
    arr = []
    for i in range(x):
        arr_2 = []
        for j in range(x):
            arr_2.append(1)
        arr.append(arr_2)
    return (1 / (x*x)) * np.array(arr)   
######################################################################################################

################################# 2. Median Filtering ################################################
def median_filter():
    path = "median_filters/"
    create_folder(path)
    #image names
    image1_filename = "image1.png"
    image2_filename = "image2.png"
    #read the files
    image1 = Image.open(image1_filename)
    image2 = Image.open(image2_filename)    
    #convert image to actual array
    #3x3,5x5,7x7 for image 1
    image1_arr_3 = np.array(image1, dtype=np.uint8)
    image1_arr_5 = np.array(image1, dtype=np.uint8)
    image1_arr_7 = np.array(image1, dtype=np.uint8)
    #3x3,5x5,7x7 for image 2
    image2_arr_3 = np.array(image2, dtype=np.uint8)
    image2_arr_5 = np.array(image2, dtype=np.uint8)
    image2_arr_7 = np.array(image2, dtype=np.uint8)
    #pad the image arrays
    #3x3: 1 top 1 bottom 1 left 1 right
    #example
    #   0 0 0 0 0
    #   0 1 2 3 0
    #   0 4 5 6 0
    #   0 0 0 0 0
    #5x5: 2 top 2 bottom 2 left 2 right
    #7x7: 3 top 3 bottom 3 left 3 right
    np.pad(image1_arr_3, ((1,1),(1,1)), 'constant')
    np.pad(image1_arr_5, ((2,2),(2,2)), 'constant')
    np.pad(image1_arr_7, ((3,3),(3,3)), 'constant')
    np.pad(image2_arr_3, ((1,1),(1,1)), 'constant')
    np.pad(image2_arr_5, ((2,2),(2,2)), 'constant')
    np.pad(image2_arr_7, ((3,3),(3,3)), 'constant')
    #apply the median filter to our images then save
    #cv2 imwrite was not working? image argument was incorrect so i used .save from Pillow
    image1_3x3 = median_3(image1_arr_3)
    image1_3x3.save("./" + path + "image1_median_3x3.png")
    image1_5x5 = median_5(image1_arr_5)
    image1_5x5.save("./" + path + "image1_median_5x5.png")
    image1_7x7 = median_7(image1_arr_7)
    image1_7x7.save("./" + path + "image1_median_7x7.png")
    image2_3x3 = median_3(image2_arr_3)
    image2_3x3.save("./" + path + "image2_median_3x3.png")
    image2_5x5 = median_5(image2_arr_5)
    image2_5x5.save("./" + path + "image2_median_5x5.png")
    image2_7x7 = median_7(image2_arr_7)
    image2_7x7.save("./" + path + "image2_median_7x7.png")
#the median filter removes noise from an image, we can expect as the kernel increases the noise will dicrease proportionally
#since there was less noise in the second image, the filtered second image appears to smooth
    
# Median Filtering Functions 
#iterate through the image pixels using a 3x3 kernel
def median_3(image_arr):
    #using these ranges we will effectively iterate through our kernel like so:
    # 0, 0 -> 0, 1 -> 0, 2
    # 1, 0 -> 1, 1 -> 1, 2
    # 2, 0 -> 2, 1 -> 2, 2
    image_copy = image_arr
    for row in range(1, len(image_arr) - 1):
        for col in range(1, len(image_arr) - 1):
            arr = []
            for vert in range(-1,2):
                for horiz in range(-1,2):
                    px = image_arr[vert + row][horiz + col]
                    arr.append(px)
            arr.sort()
            new = median(arr)
            image_copy[row][col] = new
    return Image.fromarray(image_copy)

#iterate through the image pixels using a 5x5 kernel
def median_5(image_arr):
    image_copy = image_arr
    for row in range(2, len(image_arr) - 2):
        for col in range(2, len(image_arr) - 2):
            arr = []
            for vert in range(-2,3):
                for horiz in range(-2,3):
                    px = image_arr[vert + row][horiz + col]
                    arr.append(px)
            arr.sort()
            new = median(arr)
            image_copy[row][col] = new
    return Image.fromarray(image_copy)

#iterate through the image pixels using a 7x7 kernel
def median_7(image_arr):
    image_copy = image_arr
    for row in range(3, len(image_arr) - 3):
        for col in range(3, len(image_arr) - 3):
            arr = []
            for vert in range(-3,4):
                for horiz in range(-3,4):
                    px = image_arr[vert + row][horiz + col]
                    arr.append(px)
            arr.sort()
            new = median(arr)
            image_copy[row][col] = new
    return Image.fromarray(image_copy)
####################################################################################################### 

################################# 3. Gaussian Filtering ###############################################
def gaussian_filter():
    path = "gaussian_filters/"
    create_folder(path)
    image1_filename = "image1.png"
    image2_filename = "image2.png"
    #create our kernels
    kernel_3 = create_gaussian_kernel(3)
    kernel_5 = create_gaussian_kernel(5)
    kernel_10 = create_gaussian_kernel(10)
    #read the images
    image1 = cv2.imread(image1_filename,0)
    image2 = cv2.imread(image2_filename,0)
    #convolve image 1 with the kernel, constant fills the edges with the cval
    image1_3 = ndimage.convolve(image1, kernel_3, mode='constant', cval=0.0)
    image1_5 = ndimage.convolve(image1, kernel_5, mode='constant', cval=0.0)
    image1_10 = ndimage.convolve(image1, kernel_10, mode='constant', cval=0.0)
    #convolve image 2 with the kernel
    image2_3 = ndimage.convolve(image2, kernel_3, mode='constant', cval=0.0)
    image2_5 = ndimage.convolve(image2, kernel_5, mode='constant', cval=0.0)
    image2_10 = ndimage.convolve(image2, kernel_10, mode='constant', cval=0.0)
    #write the images
    cv2.imwrite("./" + path + "image1_gaussian_3.png",image1_3)
    cv2.imwrite("./" + path + "image1_gaussian_5.png",image1_5)
    cv2.imwrite("./" + path + "image1_gaussian_10.png",image1_10)
    cv2.imwrite("./" + path + "image2_gaussian_3.png",image2_3)
    cv2.imwrite("./" + path + "image2_gaussian_5.png",image2_5)
    cv2.imwrite("./" + path + "image2_gaussian_10.png",image2_10)
    

#calculate our gaussian function based on the sigma value
def gaussian_function(u, v, sigma):
    v_2 = v**2
    u_2 = u**2
    sigma_2 = sigma**2
    f1 = (1 / (2 * math.pi * sigma_2))
    exp = (v_2 + u_2) / (2 * sigma_2) * -1
    return f1 * (math.e ** exp)

#create array for our kernel, as such:
#   h(-1,-1)    h(-1,-1)    h(-1,-1) 
#   h(0,-1)     h(0,-1)     h(0,-1) 
#   h(1,-1)     h(1,0)      h(1,1) 
def create_gaussian_kernel(size):
    arr = []
    for i in range(-1,2):
        inner_arr = []
        for j in range(-1,2):
            point = gaussian_function(i, j, size)
            inner_arr.append(point)
        arr.append(inner_arr)
    return arr
# The Gaussian filter uses a gaussian function with a determined sigma to convolve the input image.
# The Gaussian filter is helpful when reducing noise, reducing detail and adding blur.
# As the standard deviation, sigma, used in the gaussian fuction increases, the reduction of noise and detail also increase, as does blur.
###################################################################################################

################################# 4. Gradient Operations ##############################################
def gradient_operation():
    # Backward difference 
    # [-1 1]
    # Forward difference 
    # [1 -1]
    # Central difference
    # [-1 0 1]
    #create our output folder
    path = "directional_gradients/"
    create_folder(path)
    image3_filename = "image3.png"
    #read the image
    image3 = cv2.imread(image3_filename,1)
    #create kernels for directionals
    #forward
    forward_x = np.asarray([[1,0,-1],[1,0,-1],[1,0,-1]]) 
    forward_y = np.asarray([[1,1,1],[0,0,0],[-1,-1,-1]]) 
    forward_x = forward_x[:,:, None]
    forward_y = forward_y[:,:, None]
    #backward - flip kernels created for forward gradient, flip on x axis for fx and y axis for fy
    backward_x = np.asarray([[-1,0,1],[-1,0,1],[-1,0,1]]) 
    backward_y = np.asarray([[-1,-1,-1],[0,0,0],[1,1,1]]) 
    backward_x = backward_x[:,:, None]
    backward_y = backward_y[:,:, None]
    #apply the directional gradients
    image_fx, image_fy = forward_backward_gradient(forward_x, forward_y, image3, path, "forward")
    image_bx, image_by = forward_backward_gradient(backward_x, backward_y, image3, path, "backward")
    central_gradient(image_fx, image_fy, image_bx, image_by, path)

def forward_backward_gradient(direction_x, direction_y, image, path, direction):
    image_x = cv2.filter2D(image, cv2.CV_64F, direction_x)
    image_y = cv2.filter2D(image, cv2.CV_64F, direction_y)
    #hypotenuse of x and y images
    image_mag = np.hypot(image_x, image_y)
    cv2.imwrite("./" + path + "image3_" + direction + "X.png",image_x)
    cv2.imwrite("./" + path + "image3_" + direction + "Y.png",image_y)
    cv2.imwrite("./" + path + "image3_" + direction + ".png",image_mag)
    return image_x, image_y

def central_gradient(image_fx, image_fy, image_bx, image_by, path):
    magnitude_x = np.hypot(image_fx, image_by)
    magnitude_y = np.hypot(image_bx, image_fy)
    cv2.imwrite("./" + path + "image3_central_fx_by.png",magnitude_x)
    cv2.imwrite("./" + path + "image3_central_bx_fy.png",magnitude_y)
#the gradient operations that are being used as filters, find the gradient or change in the tonal intensity
#the forward gradient in both axis directions is a mirror image of the backward gradient 
#the central gradient takes the hypotenuse of the forward x axis and the backward y axis, as well as forward y and backward x
####################################################################################################

################################# 5. Sobel Filtering ##################################################
def sobel_filter():
    #create our output folder
    path = "sobel_filters/"
    create_folder(path)
    image1_filename = "image1.png"
    image2_filename = "image2.png"
    #read the images, greyscale
    image1 = cv2.imread(image1_filename,0)
    image2 = cv2.imread(image2_filename,0)
    #kernels, x and y
    sobelKernel_x = np.asarray([[-1,-2,-1], [0,0,0], [1,2,1]])
    sobelKernel_y = np.asarray([[-1,0,1], [-2,0,2], [-1,0,1]])
    #convolve the images with the sobel kernels
    #gx kernel
    image1_x = ndimage.convolve(image1, sobelKernel_x, mode='constant', cval=0.0)
    image2_x = ndimage.convolve(image2, sobelKernel_x, mode='constant', cval=0.0)
    #gy kernel
    image1_y = ndimage.convolve(image1, sobelKernel_y, mode='constant', cval=0.0)
    image2_y = ndimage.convolve(image2, sobelKernel_y, mode='constant', cval=0.0)
    #get the magnitude of the images
    image1_sobel = ((image1_x**2) + (image1_y**2)) ** (1/2)
    image2_sobel = ((image2_x**2) + (image2_y**2)) ** (1/2)
    #write the images
    cv2.imwrite("./" + path + "image1_sobel_horizontal.png", image1_x)
    cv2.imwrite("./" + path + "image1_sobel_vertical.png", image1_y)
    cv2.imwrite("./" + path + "image2_sobel_horizontal.png", image2_x)
    cv2.imwrite("./" + path + "image2_sobel_vertical.png", image2_y)
    cv2.imwrite("./" + path + "image1_sobel_magnitude.png", image1_sobel)
    cv2.imwrite("./" + path + "image2_sobel_magnitude.png", image2_sobel)
#our sobel filter emphasizes the edges in our input images
#in our output images, the sobel filter emphasizes the horizontal edges as well as the vertical edges
#additonally, the sobel filter finds the hypotenuse of the horizontal and vertical edges then produces a complete sobel processed image
#######################################################################################################

################################# 6. Fast Gaussian Filtering ##########################################
def fast_gaussian():
    path="fast_gaussian/"
    create_folder(path)
    image1_filename = "image1.png"
    image2_filename = "image2.png"
    #open images
    image1 = cv2.imread(image1_filename,0)
    image2 = cv2.imread(image2_filename,0)
    #Roberts kernels from notes
    #use a signma of 3
    #could not find kernels to use so I created my own, dividing by sum
    kernel_x = np.asarray([[gaussian_function(-1,0,3)], [gaussian_function(-2,0,3)], [gaussian_function(-1,0,3)]])
    kernel_x /= np.sum(kernel_x)
    kernel_y = np.asarray([[gaussian_function(1,2,3)], [gaussian_function(0,0,3)], [gaussian_function(-1,-2,3)]])
    kernel_y /= np.sum(kernel_y)
    #convolve image with the x kernel
    image1_x = ndimage.convolve(image1, kernel_x, mode='constant', cval=0.0)
    image2_x = ndimage.convolve(image2, kernel_x, mode='constant', cval=0.0)
    #then convolve result with y kernel
    image1_fast = ndimage.convolve(image1_x, kernel_y, mode='constant', cval=0.0)
    image2_fast = ndimage.convolve(image2_x, kernel_y, mode='constant', cval=0.0)
    #write images
    cv2.imwrite("./" + path + "image1_fast_gaussian.png", image1_fast)
    cv2.imwrite("./" + path + "image2_fast_gaussian.png", image2_fast)
#fast gaussian is more efficient as we can apply one directional gradient to an image
#then apply the next directional gradient to the already processed image
#######################################################################################################

####################################### 7. Histogram ##################################################
def histogram():
    #algorithm:
    #create an array,h, full of zeros
    #for each pixel on the image A, incremement h(A(x,y)) by 1
    path = "histograms/"
    create_folder(path)
    image4_filename = "image4.png"
    #read image in greyscale
    image4 = cv2.imread(image4_filename,0)
    #create an array, h, with zero as its elements
    #256 total colors
    h = np.zeros((256,), dtype=int)
    #get dimensions of image4
    height, width = image4.shape
    #iterate through pixels
    for i in range(height):
        for j in range(width):
            pixel = image4[i,j]
            #increment pixel frequency
            h[pixel] += 1
    #bins where our values will be contained
    bins = [64, 128, 256]
    for bin in bins:
        #create histogram with length of bin size, all 0's
        histogram = np.zeros((bin,), dtype=int)
        #find range per interval
        #i.e. 256 pixels, 64 bins, so a range of 4 pixels per interval
        interval = 256 / bin
        for i in range(len(h)):
            #add frequency of said pixel to the bin
            #index of our h divided by interval, would give us the range per bin
            #i.e. interval = 4 when bin is 64, first 4 elements will fit into range aka bin 0
            # 0 / 4 -> bin 0
            # 1 / 4 -> bin 0
            # ... 4 / 4 -> bin 1
            # 5 / 4 -> bin 1
            index = math.floor(i/interval)
            histogram[index] += h[i]
        #plot the histogram, name the labels, save as a png and clear the plot
        plt.plot(histogram)
        plt.xlabel('Black <---- Greyscale Tone ----> White')
        plt.ylabel('Pixel Frequency')
        plt.savefig("./" + path + "histogram_bin" + str(bin) + ".png")
        plt.clf()
#as we increase the bin size we get the full range of colors and more detail in each histogram
#the histograms are more representiative of the image's true color, there are more ranges thus more accuracy
#######################################################################################################

########################## 8. Can we use Entropy for thresholding? ####################################
def entropy_threshold():
    path = "entropy_threshold/"
    create_folder(path)
    image4_filename = "image4.png"
    #read image in greyscale
    image4 = cv2.imread(image4_filename,0)
    height, width = image4.shape
    #using a threshold of 10
    threshold_10 = apply_threshold(image4, 10, height, width, path)
    image4 = cv2.imread(image4_filename,0)
    #using a threshold of 100
    threshold_100 = apply_threshold(image4, 100, height, width, path)
    image4 = cv2.imread(image4_filename,0)
    #using a threshold of 150
    threshold_150 = apply_threshold(image4, 150, height, width, path)
    image4 = cv2.imread(image4_filename,0)
    #using a threshold of 200
    threshold_200 = apply_threshold(image4, 200, height, width, path)
    image4 = cv2.imread(image4_filename,0)
    #using a threshold of 250
    threshold_250 = apply_threshold(image4, 250, height, width, path)
    image4 = cv2.imread(image4_filename,0)
    

def apply_threshold(image, threshold, height, width, path):
    image_copy = image
    for i in range(0, height):
        for j in range(0, width):
            if image[i,j] > threshold:
                image_copy[i,j] = 255
            else:
                image_copy[i,j] = 0
    cv2.imwrite("./" + path + "threshold_" + str(threshold) + ".png", image_copy)
    return image_copy
# Using a specific threshold, we can turn a greyscale 0-255 image into a binary image.
# Based on the threshold, pixels with a value above it will be set to 255 (white),
# while pixels below the threshold will be set to 0 (black). Resulting in a binary image.

#######################################################################################################

################################### 9. Canny Edge Detection ###########################################
def canny_detection():
    path = "canny/"
    create_folder(path)
    #0pt Use two different images (canny1.jpg and canny2.jpg) to perform the following steps for getting Canny edgesof the input image.
    image1_filename = "canny1.jpg"
    image2_filename = "canny2.jpg"
    image1 = cv2.imread(image1_filename,1)
    image2 = cv2.imread(image2_filename,1)
    #use gaussian of 1
    image1_max_sigma1 = apply_sigma(image1, 1)
    image2_max_sigma1 = apply_sigma(image2, 1)
    cv2.imwrite("./" + path + "image1_max_sigma1.png", image1_max_sigma1)
    cv2.imwrite("./" + path + "image2_max_sigma1.png", image2_max_sigma1)

    #1pts Implement hysteresis thresholding algorithm and use it to further enhance the edge map obtained from theprevious step. 
    #Show the final Canny edge map on the screen and in the report.

    #1pt Show the effect of σ in edge detection by choosing three different σ values when smoothing. 
    #Note that youneed to indicate which σ works best as a comment in your assignment.
    #use sigma of 5
    image1_max_sigma2 = apply_sigma(image1, 5)
    image2_max_sigma2 = apply_sigma(image2, 5)
    #write images
    cv2.imwrite("./" + path + "image1_max_sigma2.png", image1_max_sigma2)
    cv2.imwrite("./" + path + "image2_max_sigma2.png", image2_max_sigma2)
    #use sigma of 10
    image1_max_sigma3 = apply_sigma(image1, 10)
    image2_max_sigma3 = apply_sigma(image2, 10)
    #write images
    cv2.imwrite("./" + path + "image1_max_sigma3.png", image1_max_sigma3)
    cv2.imwrite("./" + path + "image2_max_sigma3.png", image2_max_sigma3)

#applies the sigma, process the gradient images, then calls max suppression based on orientation and magnitude
def apply_sigma(image, sigma):
    #create the gaussian kernel
    #1 pts Use 1-dimensional Gaussians to implement 2D Gaussian filtering yourself (do not use built-in functions).
    #used the filers from fast gaussian
    gaussian = np.asarray([[gaussian_function(-1,0,sigma)], [gaussian_function(-2,0,sigma)], [gaussian_function(-1,0,sigma)]])
    gaussian /= np.sum(gaussian)
    image_copy = cv2.filter2D(image, cv2.CV_64F, gaussian)
    #1pts Obtain gradient images (x-dim, y-dim, gradient magnitude, and gradient orientation) by following the 
    #Canny algorithm that we have seen in the class. Show resulting gradient images on screen and in the report.
    #ndimage does not work for convolving 3D, need to use signal, or filter2d
    #create gradients, ndimage does not work for convolving 3D, need to use signal, or filter2d
    gradient_x = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_y = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #convert to 3d
    gradient_x = gradient_x[:,:, None]
    gradient_y = gradient_y[:,:, None]
    #x dim
    image_gradient_x = signal.convolve(image_copy, gradient_x, mode='same')
    #y dim
    image_gradient_y = signal.convolve(image_copy, gradient_y, mode='same')
    #gradient magnitude
    image_gradient_mag = np.hypot(image_gradient_x, image_gradient_y)
    #gradient orientation
    image_gradient_orient = np.arctan2(image_gradient_x, image_gradient_y)
    image_max_sigma = suppression(image_gradient_mag, image_gradient_orient)
    return image_max_sigma

#2pts Implement non-max suppression algorithm to reduce some of the falsely detected edges in the gradient images(from the previous step).
#Show the improved edge map on the screen and in the report.
def suppression(image, img_arctan):
    height, width, channel = image.shape
    image_copy = np.zeros((height,width,channel))
    #iterate through the 3d image matrix
    for i in range(1,height - 1):
        for j in range(1,width - 1):
            for k in range(channel):
                angle = np.rad2deg(img_arctan[i,j][k]) % 180
                #angle within this range results in 0
                if (0 <= angle and angle < 22.5) or (157.5 <= angle and angle < 180):
                    if (image[i, j][k] >= image[i, j - 1][k]) and (image[i, j][k] >= image[i, j + 1][k]):
                        image_copy[i,j] = image[i,j][k]
                #angle within this range results in 45
                elif (22.5 <= angle < 67.5):
                    if (image[i, j][k] >= image[i - 1, j + 1][k]) and (image[i, j][k] >= image[i + 1, j - 1][k]):
                        image_copy[i,j][k] = image[i,j][k]
                #angle within this range results in 90
                elif (67.5 <= angle < 112.5):
                    if (image[i, j][k] >= image[i - 1, j][k]) and (image[i, j][k] >= image[i + 1, j][k]):
                        image_copy[i,j][k] = image[i,j][k]
                #angle within this range results in 135
                elif (112.5 <= angle < 157.5):
                    if (image[i, j][k] >= image[i - 1, j - 1][k]) and (image[i, j][k] >= image[i + 1, j + 1][k]):
                        image_copy[i,j][k] = image[i,j][k]
    return image_copy
# The Canny Edge Detection tool uses a series of algorithms to detect in depth information in the image.
# Using said series of algorithms, the Canny operator can detect a wide range of edges within the image.
# Canny takes into account our gaussian filtering to smooth and blur the input image. Once the image has
# been blurred, we then apply our gradient operations to the image, to ultimately process an image magnitude with a specific sigma.
# Canny seems to work optimally when a median standard deviation is used, no outliers, not too low nor too high.
# We then use suppression to remove anomalies occuring from edge detection. We then use a threshold to determine genuine edges and track by hysteresis.
#######################################################################################################


if __name__ == "__main__":
    main()



        