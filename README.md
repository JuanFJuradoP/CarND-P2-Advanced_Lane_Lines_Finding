# **CarND-P2 - Advanced Lane Finding**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

One of the most important characteristics for autonomous cars is the efficient detection of lines on highways. Computer vision methodsare used with high frequency. This project shows a line detection using computer vision algorithms robustly. (Computer Vision Fundamentals, Camera Calibration, Gradients and Color Spaces, and advanced computer vision from Udacity's Self driving car Nano degree program).

![Lanes Image](./examples/example_output.jpg)

## **Getting Started**

The following instructions are used to run the project and obtain the results proposed in the requirements mentioned in the document. These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

This project has just one script on python called pipeline.py

When the script is executed in python, the resulting video will be obtained with the frame-by-frame process with the data shown on the screen.

### **Prerequisites**

To execute the advanced line detection algorithm, you must have the following requirements that, according to the operating system, must be installed.
```
Python 2.7.
OpenCV 3.0.0.
Ubuntu 16.04 (Recomendado)
```

## **Running the code**

Once the prerequisites have been installed, a cloning of the repository should be done in Github.

```
git clone  https://github.com/JuanFJuradoP/CarND-P2-Advanced_Lane_Lines_Finding.git
```
Once the the repository have been cloned, it is time to execute the script in python.

```
python pipeline.py
```

## **Goals**

The main objectives of this projects are described below.

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## **Content**
The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  

The output images are stored in the folder called `output_images`. The video called `output.avi` is the final video.  

## **Authors**

* **Juan Francisco Jurado Paez**
* Phone: +1 513 909 4704 / +57 313 247 4186.
* Mail: juanjuradop@gmail.com - jj@kiwicampus.com 
* LinkedIn: https://www.linkedin.com/in/juanfjuradop/

## **License**

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details