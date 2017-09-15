#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:24:55 2017

@author: neelbakshi
"""

import numpy as np

class NeuralNetwork:
    def __init__(self,structure):
        #if structure is [4,5,3]
        #it means that there are four input nodes, 5 nodes in hidden layer 1
        #and 3 output nodes
        self.structure = structure
        self.weights = [np.random.rand(num_of_nodes_prev, ) \
                        for num_of_nodes_curr, num_of_nodes_prev in \
                        zip(structure[1:], structure[:-1])]
        self.biases = [np.random.rand(num_of_nodes_curr, 1) \
                       for num_of_nodes_curr in structure[1:]]
        
    def activation(self, x):
        #using the sigmoid function
        return 1/(1+np.exp(-x))
    
    def activation_derivative(self, x):
        return self.activation(x)*(1 - self.activation(x))
    
    def cost_function_derivate(self, predicted_output, actual_output):
        return predicted_output - actual_output
    
    def layer_output(self, input_to_layer, layer_weights, layer_bias):
        #lets say this hidden layer has 150 nodes
        #input layer has 200 nodes
        #batch size is 500
        #then input.shape = (500, 200)
        #hidden layer weights will be of the shape (200, 150)
        
        multiply_weights_product= np.dot(input_to_layer, layer_weights)
        output = np.zeros_like(multiply_weights_product)
        #multiply_weights_product.shape = (500,150)
        #output.shape = (500,150)
        #layer_bias.shape = (150, )
        
        #we take out each input from the batch and add the biases to it
        for index, product in enumerate(multiply_weights_product):
                output[index] = self.activation(product+layer_bias)
        
        return output
    
    def feed_forward(self, input_batch):
        
    
    def fit(self, X, Y, learning_rate, epochs):
        self.learning_rate = learning_rate
                
            
    
    
        