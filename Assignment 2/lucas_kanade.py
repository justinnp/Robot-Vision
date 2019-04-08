import cv2
import os
import numpy as np
from PIL import Image
from scipy import signal

# Justin Powell
# UCF - CAP4453
# Spring 2019 

#Main Function
#Reads the images, creates the output paths and calls the LK functions
def main():
    #file names for the images
    image1_filename = "./basketball1.png"
    image2_filename = "./basketball2.png"

    #read the images into a variable
    frame1 = cv2.imread(image1_filename, 0)
    frame2 = cv2.imread(image2_filename, 0)

    #lucas kanade optical flow for image1 and image 2
    #create a folder where our single resolution lucas outputs will be stored
    path1 = './Lucas Kanade Outputs/'
    create_folder(path1)

    #run lucas kanade for single resolution
    print()
    print('Running Single Resolution Lucas Kanade. Outputs will be stored in: ' + path1)
    print()
    lucas_kanade(frame1, frame2, path1, 'single_res_lk.png')

    #create a folder where our pyramid outputs will be stored
    path2 = './Lucas Kanade Pyramid Outputs/'
    create_folder(path2)

    #run lucas kanade for multi resolution using a 1 level pyr
    print('Running Multi Resolution Lucas Kanade with 1 level Gaussian Pyramid. Outputs will be stored in: ' + path2)
    print()
    multi_resolution(frame1, frame2, path2, 1)

    #run lucas kanade for multi resolution using a 2 level pyr
    print('Running Multi Resolution Lucas Kanade with 2 level Gaussian Pyramid. Outputs will be stored in: ' + path2)
    print()
    multi_resolution(frame1, frame2, path2, 2)

    #run lucas kanade for multi resolution using a 3 level pyr
    print('Running Multi Resolution Lucas Kanade with 3 level Guassian Pyramid. Outputs will be stored in: ' + path2)
    print()
    multi_resolution(frame1, frame2, path2, 3)

    #run lucas kanade for multi resolution using a 4 level pyr
    print('Running Multi Resolution Lucas Kanade with 4 level Gaussian Pyramid. Outputs will be stored in: ' + path2)
    print()
    multi_resolution(frame1, frame2, path2, 4)

#Directory Creation Function
#function to create the path where our images will be stored
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


######################## Single Resolution Lucas Kanade Optical Flow #################################### 

#Implement Lucas-Kanade optical flow estimation, and test it for the
#two-frame data set provided in the webcourses: basketball.

def lucas_kanade(frame1, frame2, path, filename, corners=100):
    # params for ShiTomasi corner detection
    #from the sample code provided
    feature_params = dict( maxCorners = corners,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # # Parameters for lucas kanade optical flow
    # #from the sample code provided
    # lk_params = dict( winSize  = (15,15),
    #                 maxLevel = 2,
    #                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    #for our flow vectors
    #its really annoying when the color is white
    color = np.random.randint(0,255,(100,3))

    #kernels for x, y, and t dimensions
    kernel_x = np.array([[-.25, .25], [-.25, .25]]) 
    kernel_y = np.array([[-.25, -.25], [.25, .25]]) 
    kernel_t = np.array([[.25, .25], [.25, .25]])

    #calculate derivatives of dimensional kernels - fx, fy, and ft
    #convolve in each dimension
    fx = signal.convolve2d(frame1, kernel_x, mode='same') + signal.convolve2d(frame2, kernel_x, mode='same')
    fy = signal.convolve2d(frame1, kernel_y, mode='same') + signal.convolve2d(frame2, kernel_y, mode='same')
    ft = signal.convolve2d(frame1, kernel_t, mode='same') + signal.convolve2d(frame2, -kernel_t, mode='same')

    #apply a guassian blur with 0 padding and a kernel size of 3 to our frame
    #per the pdf, it is okay to use this function
    blur = cv2.GaussianBlur(frame1,(3,3),0)

    #find features/corners within the frame, use the params from the sample code and opencv's goodfeaturestotrack
    features = cv2.goodFeaturesToTrack(frame1, mask = None, **feature_params)

    #Colored copy for the colored flow vectors that will be displayed on our output
    frame_colored = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

    # Create a mask frame for drawing purposes
    frame_mask = np.zeros_like(frame_colored)

    #our vectors for optical flow
    u = np.zeros(frame1.shape)
    v = np.zeros(frame1.shape)

    #point vectors in each kernel dimension
    px = np.empty(10)
    py = np.empty(10)
    pt = np.empty(10)

    #iterate through features, adding vector heads, vector tails then eveutally append to image
    for feature in np.int0(features):
        #ravel -> flattened array returns (,) shape
        #"flattened contiguous array"
        x,y = feature.ravel()
        
        #draws vector head with radius of 2, color from our generator
        frame_colored = cv2.circle(frame_colored, (x,y), 2, color[0].tolist(), -1)

        #iterator for our for point vectors
        ix = 0

        #iterate and add our points to the point vectors
        for i in range(-1,2):
            for j in range(-1,2):
                px[ix] = fx[y + i][x + j]
                py[ix] = fy[y + i][x + j]
                pt[ix] = ft[y + i][x + j]
                ix += 1

        #create a matrix from our x and y dimensional point vectors
        matrix = np.array(np.matrix((px,py)))

        #transpose said matrix
        matrix_t = np.array(np.matrix.transpose(np.matrix((px,py))))

        #dot product of our transpose and original
        matrix_dot = np.dot(matrix, matrix_t)

        #inverse matrix, retrieve our final point, dot product of original and inversed dot
        lk = np.dot(np.linalg.inv(matrix_dot), matrix)

        #update our optical flow vectors by finding the dot product of the
        #inverse lk matrix and our t dimensional point vector
        u[y, x], v[y, x] = np.dot(lk, pt)

        #generate a vector line
        a,b = (int(v[y,x]) + x, int(u[y,x]) + y)

        #tail of the vector, will be appeneded to image
        frame_mask = cv2.line(frame_mask, (x,y),(a,b), color[0].tolist(), 2)

    #add the vector heads and tails to our image
    img = cv2.add(frame_colored, frame_mask)

    #save the image
    cv2.imwrite(path + filename, img)

############################################################################################## 
    
    
################################ Multi Resolution Lucas Kanade Optical Flow ################## 

# Implement Lucas-Kanade optical flow estimation algorithm in a multi-resolution
# Gaussian pyramid framework. You have to use your code that you developed above,
# and then you have to experimentally optimize number of levels for Gaussian pyramid,
# local window size, and Gaussian width, use the same data set (basketball) to find
# optical flows, visually compare your results with the previous step where you don’t
# use Gaussian pyramid.
  
def multi_resolution(frame1, frame2, path, scale):
    #create copies of our images
    copy1 = frame1
    copy2 = frame2

    #for loop to downsample the images based on the scale we are using
    #1,2,3,4 are hardcoded so we will have pyramids with all of those levels
    for i in range(0, scale):
        #pyrdown downsamples an image
        copy1 = cv2.pyrDown(copy1)
        copy2 = cv2.pyrDown(copy2)

    #create filename based off the scale we used
    filename = 'multi_res_lk_' + str(scale) + '.png'

    #call the lucas_kanade function with the downsampled images which will write the file
    lucas_kanade(copy1, copy2, path, filename)

##############################################################################################      


if __name__ == "__main__":
    main()
