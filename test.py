# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:59:53 2016

@author: Ari
"""

from numpy import savetxt, loadtxt, round, zeros, sin, cos, arctan2, clip, pi, tanh, exp, arange, dot, outer, array, shape, zeros_like, reshape, mean, median, max, min
from numpy.random import rand, shuffle
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

###########
# Functions
###########

# Returns a B&W image of a line represented as a binary vector of length width*height
def gen_train_image(angle, width, height, thickness):
    image = zeros((height,width))
    x_0,y_0 = width/2, height/2
    c,s = cos(angle),sin(angle)
    for y in range(height):
        for x in range(width):
            if abs((x-x_0)*c + (y-y_0)*s) < thickness/2 and -(x-x_0)*s + (y-y_0)*c > 0:
                image[x,y] = 1
    return image.flatten()

# Display training image    
def display_image(image,height, width):    
    img = plt.imshow(reshape(image,(height,width)), interpolation = 'nearest', cmap = "Greys")
    plt.show()    

# Activation function
def sigmoid(X):
    return 1.0/(1+exp(-clip(X,-50,100)))

# Returns encoded angle using specified method ("binned","scaled","cossin","gaussian")
def encode_angle(angle, method):
    if method == "binned": # 1-of-500 encoding
        X = zeros(500)
        X[int(round(250*(angle/pi + 1)))%500] = 1
    elif method == "gaussian": # Leaky binned encoding
        X = array([i for i in range(500)])
        idx = 250*(angle/pi + 1)
        X = exp(-pi*(X-idx)**2)
    elif method == "scaled": # Scaled to [-1,1] encoding
        X = array([angle/pi])
    elif method == "cossin": # Oxinabox's (cos,sin) encoding
        X = array([cos(angle),sin(angle)])
    else:
        pass
    return X

# Returns decoded angle using specified method
def decode_angle(X, method):
    if method == "binned" or method == "gaussian": # 1-of-500 or gaussian encoding
        M = max(X)
        for i in range(len(X)):
            if abs(X[i]-M) < 1e-5:
                angle = pi*i/250 - pi
                break
#        angle = pi*dot(array([i for i in range(500)]),X)/500  # Averaging
    elif method == "scaled": # Scaled to [-1,1] encoding
        angle = pi*X[0]
    elif method == "cossin": # Oxinabox's (cos,sin) encoding
        angle = arctan2(X[1],X[0])
    else:
        pass
    return angle

# Train and test neural network with specified angle encoding method
def test_encoding_method(train_images,train_angles,test_images, test_angles, method, num_iters, alpha = 0.01, alpha_bias = 0.0001, momentum = 0.9, hid_layer_size = 500):
    num_train,in_layer_size = shape(train_images)
    num_test = len(test_angles)

    if method == "binned":
        out_layer_size = 500
    elif method == "gaussian":
        out_layer_size = 500
    elif method == "scaled":
        out_layer_size = 1
    elif method == "cossin":
        out_layer_size = 2
    else:
        pass

    # Initial weights and biases
    IN_HID = rand(in_layer_size,hid_layer_size) - 0.5 # IN --> HID weights
    HID_OUT = rand(hid_layer_size,out_layer_size) - 0.5 # HID --> OUT weights
    BIAS1 = rand(hid_layer_size) - 0.5 # Bias for hidden layer
    BIAS2 = rand(out_layer_size) - 0.5 # Bias for output layer


    # Train
    for j in range(num_iters):
        for i in range(num_train):
            # Get training example
            IN = train_images[i]
            TARGET = encode_angle(train_angles[i],method) 

            # Feed forward and compute error derivatives
            HID = sigmoid(dot(IN,IN_HID)+BIAS1)

            if method == "binned" or method == "gaussian": # Use softmax
                OUT = exp(clip(dot(HID,HID_OUT)+BIAS2,-100,100))
                OUT = OUT/sum(OUT)
                dACT2 = OUT - TARGET
            elif method == "cossin" or method == "scaled": # Linear
                OUT = dot(HID,HID_OUT)+BIAS2 
                dACT2 = OUT-TARGET 
            else:
                print("Invalid encoding method")

            dHID_OUT = outer(HID,dACT2)
            dACT1 = dot(dACT2,HID_OUT.T)*HID*(1-HID)
            dIN_HID = outer(IN,dACT1)
            dBIAS1 = dACT1
            dBIAS2 = dACT2


            # Update the weights
            HID_OUT -= alpha*dHID_OUT
            IN_HID -= alpha*dIN_HID
            BIAS1 -= alpha_bias*dBIAS1
            BIAS2 -= alpha_bias*dBIAS2

    # Test
    test_errors = zeros(num_test)
    angles = zeros(num_test)
    target_angles = zeros(num_test)
    accuracy_to_point001 = 0
    accuracy_to_point01 = 0
    accuracy_to_point1 = 0

    for i in range(num_test):

        # Get training example
        IN = test_images[i]
        target_angle = test_angles[i]

        # Feed forward
        HID = sigmoid(dot(IN,IN_HID)+BIAS1)

        if method == "binned" or method == "gaussian":
            OUT = exp(clip(dot(HID,HID_OUT)+BIAS2,-100,100))
            OUT = OUT/sum(OUT)
        elif method == "cossin" or method == "scaled":
            OUT = dot(HID,HID_OUT)+BIAS2 

        # Decode output 
        angle = decode_angle(OUT,method)

        # Compute errors
        error = abs(angle-target_angle)
        test_errors[i] = error
        angles[i] = angle

        target_angles[i] = target_angle
        if error < 0.1:
            accuracy_to_point1 += 1
        if error < 0.01: 
            accuracy_to_point01 += 1
        if error < 0.001:
            accuracy_to_point001 += 1

    # Compute and return results
    accuracy_to_point1 = 100.0*accuracy_to_point1/num_test
    accuracy_to_point01 = 100.0*accuracy_to_point01/num_test
    accuracy_to_point001 = 100.0*accuracy_to_point001/num_test

    return mean(test_errors),median(test_errors),min(test_errors),max(test_errors),accuracy_to_point1,accuracy_to_point01,accuracy_to_point001

# Dispaly results
def display_results(results,method):
    MEAN,MEDIAN,MIN,MAX,ACC1,ACC01,ACC001 = results
    if method == "binned":
        print("Test error for 1-of-500 encoding:")
    elif method == "gaussian":
        print("Test error for gaussian encoding: ")
    elif method == "scaled":
        print("Test error for [-1,1] encoding:")
    elif method == "cossin":
        print("Test error for (cos,sin) encoding:")
    else:
        pass
    print("-----------")
    print("Mean: "+str(MEAN))
    print("Median: "+str(MEDIAN))
    print("Minimum: "+str(MIN))
    print("Maximum: "+str(MAX))
    print("Accuracy to 0.1: "+str(ACC1)+"%")
    print("Accuracy to 0.01: "+str(ACC01)+"%")
    print("Accuracy to 0.001: "+str(ACC001)+"%")
    print("\n\n")


##################
# Image parameters
##################
width = 100 # Image width
height = 100 # Image heigth
thickness = 5.0 # Line thickness

#################################
# Generate training and test data
#################################
num_train = 1000
num_test = 1000
test_images = []
test_angles = []
train_images = []
train_angles = []
for i in range(num_train):
    angle = pi*(2*rand() - 1)
    train_angles.append(angle)
    image = gen_train_image(angle,width,height,thickness)
    train_images.append(image)
for i in range(num_test):
    angle = pi*(2*rand() - 1)
    test_angles.append(angle)
    image = gen_train_image(angle,width,height,thickness)
    test_images.append(image)
train_angles,train_images,test_angles,test_images = array(train_angles),array(train_images),array(test_angles),array(test_images)



###########################
# Evaluate encoding schemes
###########################
num_iters = 50

# Train with cos,sin encoding
method = "cossin"
results1 = test_encoding_method(train_images, train_angles, test_images, test_angles, method, num_iters)
display_results(results1,method)

# Train with scaled encoding
method = "scaled"
results3 = test_encoding_method(train_images, train_angles, test_images, test_angles, method, num_iters)
display_results(results3,method)

# Train with binned encoding
method = "binned"
results2 = test_encoding_method(train_images, train_angles, test_images, test_angles, method, num_iters)
display_results(results2,method)

# Train with gaussian encoding
method = "gaussian"
results4 = test_encoding_method(train_images, train_angles, test_images, test_angles, method, num_iters)
display_results(results4,method)
