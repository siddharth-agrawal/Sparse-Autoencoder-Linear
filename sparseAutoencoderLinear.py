# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2014 Siddharth Agrawal
# Code written by : Siddharth Agrawal
# Email ID : siddharth.950@gmail.com

import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot

###########################################################################################
""" The Sparse Autoencoder Linear class """

class SparseAutoencoderLinear(object):

    #######################################################################################
    """ Initialization of Autoencoder object """

    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
    
        """ Initialize parameters of the Autoencoder object """
    
        self.visible_size = visible_size    # number of input units
        self.hidden_size = hidden_size      # number of hidden units
        self.rho = rho                      # desired average activation of hidden units
        self.lamda = lamda                  # weight decay parameter
        self.beta = beta                    # weight of sparsity penalty term
        
        """ Set limits for accessing 'theta' values """
        
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size
        
        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """
        
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        
        rand = numpy.random.RandomState(int(time.time()))
        
        W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        
        """ Bias values are initialized to zero """
        
        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        """ Create 'theta' by unrolling W1, W2, b1, b2 """

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    #######################################################################################
    """ Returns elementwise sigmoid output of input array """
    
    def sigmoid(self, x):
    
        return (1 / (1 + numpy.exp(-x)))

    #######################################################################################
    """ Returns the cost of the Autoencoder and gradient at a particular 'theta' """
        
    def sparseAutoencoderLinearCost(self, theta, input):
        
        """ Extract weights and biases from 'theta' input """
        
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        
        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """
        
        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = numpy.dot(W2, hidden_layer) + b2
        
        """ Estimate the average activation value of the hidden layers """
        
        rho_cap = numpy.sum(hidden_layer, axis = 1) / input.shape[1]
        
        """ Compute intermediate difference values using Backpropagation algorithm """
        
        diff = output_layer - input
        
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]
        weight_decay         = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)))
        KL_divergence        = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                                    (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))
        cost                 = sum_of_squares_error + weight_decay + KL_divergence
        
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        
        del_out = diff
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)), 
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))
        
        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """
            
        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis = 1)
        b2_grad = numpy.sum(del_out, axis = 1)
            
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        
        """ Transform numpy matrices into arrays """
        
        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)
        
        """ Unroll the gradient values and return as 'theta' gradient """
        
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))
                                        
        return [cost, theta_grad]

###########################################################################################
""" Preprocesses the dataset using ZCA Whitening """

def preprocessDataset(data, num_patches, epsilon):

    """ Subtract mean of each patch separately """

    mean_patch = numpy.mean(data, axis = 1, keepdims = True)
    data       = data - mean_patch
    
    """ Compute the ZCA Whitening matrix """
    
    sigma           = numpy.dot(data, numpy.transpose(data)) / num_patches
    [u, s, v]       = numpy.linalg.svd(sigma)
    rescale_factors = numpy.diag(1 / numpy.sqrt(s + epsilon))
    zca_white       = numpy.dot(numpy.dot(u, rescale_factors), numpy.transpose(u));
    
    """ Apply ZCA Whitening to the data """
    
    data = numpy.dot(zca_white, data)
    
    return data, zca_white, mean_patch

###########################################################################################
""" Loads the image patches from the mat file """

def loadDataset():

    """ Loads the dataset as a numpy array
        The dataset is originally read as a dictionary """

    images = scipy.io.loadmat('stlSampledPatches.mat')
    images = numpy.array(images['patches'])
    
    return images
    
###########################################################################################
""" Visualizes the obtained optimal W1 values as images """

def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):

    """ Add the weights as a matrix of images """
    
    figure, axes = matplotlib.pyplot.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    
    """ Rescale the values from [-1, 1] to [0, 1] """
    
    opt_W1 = (opt_W1 + 1) / 2
    
    """ Define useful values """
    
    index  = 0
    limit0 = 0
    limit1 = limit0 + vis_patch_side * vis_patch_side
    limit2 = limit1 + vis_patch_side * vis_patch_side
    limit3 = limit2 + vis_patch_side * vis_patch_side
                                              
    for axis in axes.flat:
    
        """ Initialize image as array of zeros """
    
        img = numpy.zeros((vis_patch_side, vis_patch_side, 3))
        
        """ Divide the rows of parameter values into image channels """
        
        img[:, :, 0] = opt_W1[index, limit0 : limit1].reshape(vis_patch_side, vis_patch_side)
        img[:, :, 1] = opt_W1[index, limit1 : limit2].reshape(vis_patch_side, vis_patch_side)
        img[:, :, 2] = opt_W1[index, limit2 : limit3].reshape(vis_patch_side, vis_patch_side)
        
        """ Plot the image on the figure """
        
        image = axis.imshow(img, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
        
    """ Show the obtained plot """  
        
    matplotlib.pyplot.show()

###########################################################################################
""" Loads data, trains the Autoencoder and visualizes the learned weights """

def executeSparseAutoencoderLinear():

    """ Define the parameters of the Autoencoder """
    
    image_channels  = 3      # number of channels in the image patches
    vis_patch_side  = 8      # side length of sampled image patches
    hid_patch_side  = 20     # side length of representative image patches
    num_patches     = 100000 # number of training examples
    rho             = 0.035  # desired average activation of hidden units
    lamda           = 0.003  # weight decay parameter
    beta            = 5      # weight of sparsity penalty term
    max_iterations  = 400    # number of optimization iterations
    epsilon         = 0.1    # regularization constant for ZCA Whitening
    
    visible_size = vis_patch_side * vis_patch_side * image_channels # number of input units
    hidden_size  = hid_patch_side * hid_patch_side                  # number of hidden units
    
    """ Load the dataset and preprocess using ZCA Whitening """
    
    training_data = loadDataset()
    training_data, zca_white, mean_patch = preprocessDataset(training_data, num_patches, epsilon)
    
    """ Initialize the Autoencoder with the above parameters """
    
    encoder = SparseAutoencoderLinear(visible_size, hidden_size, rho, lamda, beta)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    
    opt_solution  = scipy.optimize.minimize(encoder.sparseAutoencoderLinearCost, encoder.theta, 
                                            args = (training_data,), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations})
    opt_theta     = opt_solution.x
    opt_W1        = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size, visible_size)
    
    """ Visualize the obtained optimal W1 weights """
    
    visualizeW1(numpy.dot(opt_W1, zca_white), vis_patch_side, hid_patch_side)

executeSparseAutoencoderLinear()
