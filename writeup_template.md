# **CarND-P2 - Advanced Lane Finding**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

One of the most important characteristics for autonomous cars is the efficient detection of lines on highways. Computer vision methodsare used with high frequency. This project shows a line detection using computer vision algorithms robustly. (Computer Vision Fundamentals, Camera Calibration, Gradients and Color Spaces, and advanced computer vision from Udacity's Self driving car Nano degree program).

![Lanes Image](./examples/example_output.jpg)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
---

[//]: # (Image References)

[image1]: ./output_images/Readme_images/1.png "Calibration Cameras 1"

[image2]: ./output_images/Readme_images/2.png "Thresholded binary image 1"
[image3]: ./output_images/Readme_images/3.png "Thresholded binary image 2"
[image4]: ./output_images/Readme_images/4.png "Thresholded binary image 3"
[image5]: ./output_images/Readme_images/5.png "Thresholded binary image 4"
[image6]: ./output_images/Readme_images/6.png "Thresholded binary image 5"
[image7]: ./output_images/Readme_images/7.png "Thresholded binary image 6"
[image8]: ./output_images/Readme_images/8.png "Thresholded binary image 7"
[image9]: ./output_images/Readme_images/9.png "Thresholded binary image 8"
[image10]: ./output_images/Readme_images/10.png "Polynomial Fitting 1"
[image11]: ./output_images/Readme_images/11.png "Polynomial Fitting 2"
[image12]: ./output_images/Readme_images/12.png "Polynomial Fitting 3"
[image13]: ./output_images/Readme_images/13.png "Road area detection 1"
[image14]: ./output_images/Readme_images/14.png "Road area detection 2"
[image15]: ./output_images/Readme_images/15.png "Road area detection 3"
[image16]: ./output_images/Readme_images/16.png "Road area detection 4"

## **Pipeline description**
### **1. Camera Calibration**
Camera calibration is the process of estimating intrinsic and/or extrinsic parameters. Intrinsic parameters deal with the camera's internal characteristics, such as, its focal length, skew, distortion, and image center. Extrinsic parameters describe its position and orientation in the world. Knowing intrinsic parameters is an essential first step for 3D computer vision, as it allows you to estimate the scene's structure in Euclidean space and removes lens distortion, which degrades accuracy. 

Two functions are important to camera calibration. The first one is `camera_calibration()`. This function is in charge calibrate the camera according to chessboard images, returns the mtx matrix,dist and images list with points drawn. It is supossed that the image is in the plane (x,y), thus z=0.

On the other hand, I used `check_calibration_process()` function to verify and count how many images were used to calibrate the camera. Because, some images does not have the minimum amount of black squares to calibrate the camera. So, this function verify and report how many images were useful.

Find below an example of an image to show both images. Left image is a distored pattern and the right one is the undistored image (after camera calibration process).

![image1]
*Figure 1 - Distored and undistored image.*

### **2. Thresholded binary image.**

Para encontrar una imagen binaria se uso el operador Sobel para las imagenes usando la funcion `abs_sobel_thresh()`. La funcion convierte la imagen RGB en escala de grises usando la funcion `cv2.cvtColor()`. Luego, se obtiene la derivada en 'x' y 'y' respectivamente para tomar el valor absoluto de la derivada en 'x' y 'y' aplicando la funcion ` cv2.Sobel()`. 
Luego se aplico mascaras de color para determinar el punto en donde se veia las lineas blancas y amarillas usando `binary_output[scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1`.

Para tener una deteccion de lineas mas robusta se calcula la magnitud del gradiente de la imagen usando la funcion `mag_thresh()` siguiendo el siguiente pipeline. Primero se debe obtener una imagen en un solo canal, en este caso, escala de grises. Luego, se calcula los gradientes en 'X' y 'Y' por separado usando `sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)` y `sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)` respectivamente.

A continuacion se muestra algunas imagenes con el resultado obtenido.
![image2]
*Figure 2 - Thresholded binary image.*
![image3]
*Figure 3 - Thresholded binary image.*
![image4]
*Figure 4 - Thresholded binary image.*
![image5]
*Figure 5 - Thresholded binary image.*
![image6]
*Figure 6 - Thresholded binary image.*
![image7]
*Figure 7 - Thresholded binary image.*
![image8]
*Figure 8 - Thresholded binary image.*
![image9]
*Figure 9 - Thresholded binary image.*


### **3. Polynomial Fitting and Line Curvature.**

![image10]
![image11]
![image12]
### **5. Results.**
![image13]
![image14]
![image15]
![image16]

---
## **Authors**
* **Juan Francisco Jurado Paez**
* Phone: +1 513 909 4704 / +57 313 247 4186.
* Mail: juanjuradop@gmail.com - jj@kiwicampus.com 
* LinkedIn: https://www.linkedin.com/in/juanfjuradop/