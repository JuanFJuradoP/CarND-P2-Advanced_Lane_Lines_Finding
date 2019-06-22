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





























# ===================================================================================
# PIPELINE EJECUTION.
# ===================================================================================

"""
*************************************************************************************
CAMERA CALIBRATION - Parameters.
*************************************************************************************
"""

# Read in and make a list of calibration images
# Aqui tengo la ruta de las imagenes de calibracion
calibration_images_names = glob.glob('camera_cal/calibration*.jpg')

#En calibration_images_names tengo la ruta de las imagenes de calibracion
#Imprimir la ruta de las imagenes de calibracion

#print("Calibration image root", calibration_images_names)


### FUNCTION: Function to read images according to name list - read_images()
calibration_images_distored = read_images(calibration_images_names)
# ya tengo en una lista las imagenes para calibrar la camara - calibration_images_distored

### FUNCTION: Function to read images according to name list - read_images()
test_images_distored = read_images(test_images_names)

#Ya tengo dos listas, la primer lista se llama 'calibration_images_distored' y almacena
# todas las imagenes para calibrar la camara. Y la segunda lista se llama 'test_images_distored'
# son las imagenes de testeo para probar todo el algoritmo.
# de aqui en adelante se va a trabajar con esta lista 

# A continuacion se va a describir todo el pipeline. Se aconseja trabajar las funciones para
# una sola imagen y luego extrapolar a ir cambiando imagen por imagen para hacer el procesamiento
# al fin y al cabo un video se compone de imagenes que se conoce como frames.

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

"""
*************************************************************************************
PROCESS PIPELINE.
*************************************************************************************
"""
# la lista de imagenes de test sin distorsion estan en 'test_images_undistored'

# la lista de imagenes de calibracion sin distorsion estan en 'camera_cal_undistored'


# La idea de esta funcion es que reciba una imagen y retorne la imagen binaria
# usando mascaras, etc
def find_lane_lines():
    return





# Thresholded binary images
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





"""
Yo uso esta funcion para graficar las imagenes que estan en la lista
for i in range(len(calibration_images_distored)):
    plt.imshow(calibration_images_distored[i])
    plt.show()
"""

#print(test_images_names)
### FUNCTION: Function to read images according to name list - read_images()
# calibration_images_distored = read_images(calibration_images_names)
