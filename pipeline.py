# ===================================================================================
"""
Code Information:
	Developer: Juan Jurado - JJ.
	Phone:  / +1 (513) 909 4704 / +57 (313) 247 4186.
	Mail: juanjuradop@gmail.com / jj@kiwicampus.com.
    LinkedIn: 

Description: 
    Self-Driving Car Nanodegree Program.
    Part 1: Computer Vision, Deep Learning and Sensor Fusion.
        Project # 2: Advanced Lane Finding Project.
    Objective: Write a software pipeline to identify the lane boundaries in a video 
    from a front-facing camera on a car.

Tested on: 
    python 2.7.
    OpenCV 3.0.0.
    Ubuntu 16.04.

The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of 
chessboard images. OK
2. Apply a distortion correction to raw images. OK
3. Use color transforms, gradients, etc., to create a thresholded binary image.OK
4. Apply a perspective transform to rectify binary image ("birds-eye view"). OK
5. Detect lane pixels and fit to find the lane boundary.OK
6. Determine the curvature of the lane and vehicle position with respect to center.OK
7. Warp the detected lane boundaries back onto the original image.OK
8. Output visual display of the lane boundaries and numerical estimation of lane 
curvature and vehicle position.OK
"""
# ===================================================================================
# IMPORT USEFUL PACKAGES.
# ===================================================================================
# Importing useful packages.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob
import cv2
import os

# ===================================================================================
# DEFINE CLASSES.
# ===================================================================================
# Define a class to create warnings.
class images_warning(UserWarning):
    pass

# Define a class to receive the characteristics of each line detection.
class Line():
    def __init__(self):
        # was the line detected in the last iteration?.
        self.detected = False
        # x values of the last n fits of the line.
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations.
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations.
        self.best_fit = None
        #polynomial coefficients for the most recent fit.
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units.
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line.
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits.
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels.
        self.allx = None
        #y values for detected line pixels.
        self.ally = None

# ===================================================================================
# GLOBAL VARIABLES.
# ===================================================================================
# Prepare object points.
nx = 9 # Number of inside corners in x.
ny = 6 # Number of inside corners in y.

# ===================================================================================
# READING TEST IMAGES NAMES.
# ===================================================================================
# In the folder 'test_images' there is the images to test the pipeline first,
# once we have done the pipeline test using images, we could jump to test the videos.
# Making sure we import our images with the right names.

test_img_dir = "test_images/"

# I create an array with the names of all images that are inside this folder.
test_images_names = os.listdir(test_img_dir)
test_images_names = list(map(lambda name: test_img_dir + name, test_images_names))

# ===================================================================================
# FUNCTIONS.
# ===================================================================================

# OK
def read_images(names):
    # Description:
    #   Function to read images according to name list - read_images().
    #   Function that reads the name according to the root and append in a new list.
    # Inputs: 
    #   Images root names (names).
    # Outputs: 
    #   List with images according to root folder location (images).
    # Resources:
    #   N/A.

    images = []
    for i in range(len(names)):
        temp = cv2.imread(names[i])
        images.append(temp)
    return images

# OK
def camera_calibration (calibration_images_names, nx, ny):
    # Description:
    #   Function to camera calibration - camera_calibration().
    #   Calibrate the camera according to chessboard images, returns the mtx matrix, 
    #   dist and images list with points drawn.
    # Inputs: 
    #   Calibration images list (calibration_images_names), cheesboard X corners (nx)
    #   and Y corners (ny).
    # Outputs: 
    #   Camera matrix (mtx) ,distortion coefficientes (dist), images with chessboard 
    #   corners (drawChessboardCornersImages).
    # Resources:
    #   https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    # List to store Images with draw chessboard corners.
    drawChessboardCornersImages = []
    # List to store object points and image points from all the images.
    objpoints = [] # 3D points in real world space.
    # All the points has x,y,z coordinates. The z coordinate will be 0 for all the points.
    imgpoints = [] # 2D points in image plane.
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... , (7,5,0).
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates.

    for fname in calibration_images_names:
        # Read in a calibration image.
        img = cv2.imread(fname)
        # Convert to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If corners are found, add object points , image points.
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners.
            img = cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
            
            #plt.imshow(img)
            #plt.show()

            #Append each image with chessboard corners.
            drawChessboardCornersImages.append(cv2.drawChessboardCorners(img,(nx,ny),corners,ret))
    # Camera calibration, given object points, image points, and the shape of the grayscale image:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist, drawChessboardCornersImages

def check_calibration_process (drawChessboardCornersImages,calibration_images_names):
    # Description:
    #   Verify images to calibrate - check_calibration_process().
    #   This function verifies that images for calibration works according to borders on it.
    #   It means, if we send 20 images (patterns) and just work 17, send a warning that only
    #   17 patterns images calibration work for camera calibration.
    # Inputs:
    #   Chessboards images with points list.
    # Outputs:
    #   Nothing, just send a warning through console.
    # Resources:
    #   https://docs.python.org/3/library/warnings.html

    if len(drawChessboardCornersImages)!=len(calibration_images_names):
        print('You are using',len(drawChessboardCornersImages), 'of', len(calibration_images_names), 'images to calibrate your camera.' )
        warnings.warn('Not all images were used to calibrate your camera.',images_warning)
    return

def undistor_image (image, mtx, dist):
    # Description:
    #   Undistored images - undistor_image().
    #   Function that modify the iamge according to camera distorsion parameters.
    # Inputs: 
    #   distored image (image), Camera matrix (mtx), distortion coefficientes (dist).
    # Outputs: 
    #   undistored image (img).
    # Resources:
    #   N/A.

    img = cv2.undistort(image, mtx, dist, None, mtx)
    return img

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    ### FUNCTION: Magnitude of the Gradient  - mag_thresh()
    ### inputs:
    ### outputs: 
    ### Resources: 
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # 5) Create a binary mask where mag thresholds are met
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    ### FUNCTION: Direction of the Gradient  - dir_threshold()
    ### inputs:
    ### outputs: 
    ### Resources: 
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def save_images_output (direction_original, image, image_description):
    # 'direction_original' es la direccion de las imagenes originales de ahi se va a sacar el nombre,
    # 'image' es la lista de imagenes que se quiere guardar en la carpeta de salida y 'image_description' es el nombre de la
    # imagen con la que se quiere guardar, ejemplo "gray", "canny", etc.
    image_name = []
    image_names = os.listdir(direction_original)
    directory_output = '/home/juan/Desktop/Self_Driving_cars_nanodegree/Projects/CarND-P2-Advanced_Lane_Lines_Finding/output_images/'
    for i in range(len(image_names)):
        image_name.append(directory_output + image_description + image_names[i])
        #image[i].dtype='uint32'
        #image[i] = cv2.cvtColor(image[i], cv2.COLOR_GRAY2RGB)
        #image[i].astype('uint8') * 255
        #cv2.imshow("nalgas",image[i]);cv2.waitKey(0)
        cv2.imwrite(image_name[i], image[i])

        #cv2.imwrite(image[i])
    return

def radius_curvature(binary_warped, left_fit, right_fit):
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curvature =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate vehicle center
    #left_lane and right lane bottom in pixels
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    # Lane center as mid of left and right lane bottom                        
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix #Convert to meters
    position = "left" if center < 0 else "right"
    center = "Vehicle is {:.2f}m {}".format(center, position)
    
    # Now our radius of curvature is in meters
    return left_curvature, right_curvature, center

def perspective_images (img, mtx, dist, nx, ny):

    offset = 100 # offset for dst points
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    h, w = img.shape[:2]

    src = np.float32([[w, h-10],    # br
                      [0, h-10],    # bl
                      [546, 460],   # tl
                      [732, 460]])  # tr
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # Given src and dst points, calculate the perspective transform matrix    
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty

def print_list_text(img_src, str_list, origin = (0, 0), color = (0, 255, 255), thickness = 2, fontScale = 0.45,  y_space = 20):

    """  prints text list in cool way
    Args:
        img_src: `cv2.math` input image to draw text
        str_list: `list` list with text for each row
        origin: `tuple` (X, Y) coordinates to start drawings text vertically
        color: `tuple` (R, G, B) color values of text to print
        thickness: `int` thickness of text to print
        fontScale: `float` font scale of text to print
    Returns:
        img_src: `cv2.math` input image with text drawn
    """

    for idx, strprint in enumerate(str_list):
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = (0, 0, 0), 
                    thickness = thickness+3, 
                    lineType = cv2.LINE_AA)
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = color, 
                    thickness = thickness, 
                    lineType = cv2.LINE_AA)

    return img_src

def process_video (frame, mtx, dist, nx, ny):
    #Create a copy of the current frame
    original_frame = np.copy(frame)

    #Undistored image
    frame_undistort = cv2.undistort(frame, mtx, dist, None, mtx)

    #Derivate Gradx and GradY. Apply Sobel.
    gradx = abs_sobel_thresh(frame_undistort, orient='x', thresh_min=50, thresh_max=100)
    grady = abs_sobel_thresh(frame_undistort, orient='x', thresh_min=50, thresh_max=100)

    #Apply Mag and Dir Threshold
    mag_binary = mag_thresh(frame_undistort, sobel_kernel=3, mag_thresh=(40, 100))
    dir_binary = dir_threshold(frame_undistort, sobel_kernel=3, thresh=(0.7, 1.3))

    combined_frame = np.zeros_like(dir_binary)
    combined_frame[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    image = cv2.cvtColor(frame_undistort, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined_frame)
    combined_binary[(s_binary == 1) | (combined_frame == 1)] = 1

    binary_warped, M = perspective_images (combined_binary, mtx, dist, nx, ny)

    out_img, left_fit, right_fit, ploty = fit_polynomial(binary_warped)
    left_curvature, right_curvature, center = radius_curvature(out_img, left_fit, right_fit)
    text = []
    text.append(str(left_curvature))
    text.append(str(right_curvature))
    text.append(center)
    original_frame_text = print_list_text(original_frame,text, origin = (30, 50), color = (0, 255, 255), thickness = 2, fontScale = 1,  y_space = 40)
    return original_frame_text, out_img

def text_in_image (images_list,binary_warped_images):
    new_images = []
    for i in range(len(images_list)):
        out_img, left_fit, right_fit, ploty = fit_polynomial(binary_warped_images[i])
        left_curvature, right_curvature, center = radius_curvature(out_img, left_fit, right_fit)
        text_list = []
        text_list.append(str(left_curvature))
        text_list.append(str(right_curvature))
        text_list.append(center)

        image_temp = print_list_text(images_list[i],text_list, origin = (30, 50), color = (0, 255, 255), thickness = 2, fontScale = 1,  y_space = 40)
        
        new_images.append(image_temp)
        #cv2.imshow("img",new_images[i]);cv2.waitKey(0)
    return new_images

# ===================================================================================
# PIPELINE EJECUTION.
# ===================================================================================

# Read in and make a list of calibration images
calibration_images_names = glob.glob('camera_cal/calibration*.jpg')

### FUNCTION: Function to read images according to name list - read_images()
calibration_images_distored = read_images(calibration_images_names)

### FUNCTION: Function to read images according to name list - read_images()
test_images_distored = read_images(test_images_names)

### Function to camera calibration - camera_calibration()
mtx, dist, drawChessboardCornersImages = camera_calibration (calibration_images_names, nx, ny)

### Verify images to calibrate - check_calibration_process()
check_calibration_process (drawChessboardCornersImages,calibration_images_names)

test_images_undistored = []
test_images_undistored_copy = []
for i in range(len(test_images_distored)):
    image_temp = undistor_image (test_images_distored[i], mtx, dist)
    test_images_undistored.append(image_temp)
    test_images_undistored_copy.append(image_temp)

camera_cal_undistored = []
for i in range(len(calibration_images_distored)):
    image_temp = undistor_image (calibration_images_distored[i], mtx, dist)
    camera_cal_undistored.append(image_temp)

# Thresholded binary images
binary_images = []

for i in range(len(test_images_undistored)):
    gradx = abs_sobel_thresh(test_images_undistored[i], orient='x', thresh_min=50, thresh_max=100)
    grady = abs_sobel_thresh(test_images_undistored[i], orient='x', thresh_min=50, thresh_max=100)
    mag_binary = mag_thresh(test_images_undistored[i], sobel_kernel=3, mag_thresh=(40, 100))
    dir_binary = dir_threshold(test_images_undistored[i], sobel_kernel=3, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    imag = cv2.cvtColor(test_images_undistored[i], cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(imag, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined)   
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
    binary_images.append(combined_binary)
    #cv2.imshow('prueba',binary_images[i])

binary_warped_images = []
for i in range(len(binary_images)):
    #Perspective images
    binary_warped, M = perspective_images (binary_images[i], mtx, dist, nx, ny)
    binary_warped_images.append(binary_warped)


images_with_text = text_in_image (test_images_undistored_copy,binary_warped_images)

#save_images_output (test_img_dir, calibration_images_distored, "Calibration_images_distored_")
#save_images_output (test_img_dir, test_images_distored, "test_images_distored_")
#save_images_output (test_img_dir, test_images_undistored, "test_images_undistored_")
#save_images_output (test_img_dir, camera_cal_undistored, "camera_cal_undistored_")
#save_images_output (test_img_dir, images_with_text, "images_with_text")

#save_images_output (test_img_dir, binary_images, "binary_images_")
#save_images_output (test_img_dir, binary_warped_images, "binary_warped_images_")

# Read the video frame-by-frame
cap = cv2.VideoCapture('/home/kiwicampus/JuanFJuradoP/CarND-P2-Advanced_Lane_Lines_Finding/test_videos/project_video.mp4')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

"""
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame_original_text, process_frame = process_video (frame, mtx, dist, nx, ny)
        out.write(cv2.resize(frame_original_text,(1280,720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
"""

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()