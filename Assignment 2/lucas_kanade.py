import cv2
import os
import numpy as np
from PIL import Image

# Justin Powell
# UCF  - CAP4453
# Spring 2019 

def main():
    #file names for the images
    image1_filename = "./basketball1.png"
    image2_filename = "./basketball2.png"
    #read the images into a variable
    frame1 = cv2.imread(image1_filename)
    frame2 = cv2.imread(image2_filename)
    #lucas kanade optical flow for image1 and image 2
    #create a folder where our lucas outputs will be stored
    path1 = './Optical_Flow_Outputs'
    create_folder(path1)
    lucas_kanade(frame1, frame2, path1)
    #create a folder where our pyramid outputs will be stored
    path2 = './Multi_Resolution_Outputs'
    create_folder(path2)
    multi_resolution(frame1, frame2, path2)
    #multi resolution lucase kanade optical flow for image1 and image2
    # multi_resolution(frame1, frame2)


#function to create the path where our images will be stored
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Implement Lucas-Kanade optical flow estimation, and test it for the
#two-frame data set provided in the webcourses: basketball.
def lucas_kanade(frame1, frame2, path):
    #create a folder where our outputs will be stored
    # params for ShiTomasi corner detection
    #from the sample code provided
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Generate random colors for the vectors
    color = np.random.randint(0,255,(100,3))

    #kernels for x y and t dimensions
    kernel_x = np.array([[-1, 1], [-1, 1]]) / 4
    kernel_y = np.array([[-1, -1], [1, 1]]) / 4
    kernel_t = np.array([[1, 1], [1, 1]]) / 4

    #calculate derivatives fxx, fy, ft
    f_x = signal.convolve2d(frame1, kernel_x, boundary='symm', mode='same')
    f_y = signal.convolve2d(frame1, kernel_y, boundary='symm', mode='same')
    f_t = signal.convolve2d(frame1, kernel_t, boundary='symm', mode='same') +
         signal.convolve2d(frame2, -kernel_t, boundary='symm', mode='same')

    #find features within the frame, use the params from the sample code
    features = cv2.goodFeaturesToTrack(frame1, mask = None, **feature_params)
    #for the colored flow vectors that will be displayed on our outputs
    frame1_colored = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    frame1_mask = np.zeros_like(frame_colored)
    

# Implement Lucas-Kanade optical flow estimation algorithm in a multi-resolution
# Gaussian pyramid framework. You have to use your code that you developed above,
# and then you have to experimentally optimize number of levels for Gaussian pyramid,
# local window size, and Gaussian width, use the same data set (basketball) to find
# optical flows, visually compare your results with the previous step where you don’t
# use Gaussian pyramid.   
def multi_resolution(frame1, frame2):


if __name__ == "__main__":
    main()
