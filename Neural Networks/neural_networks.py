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
        self.weights = [np.random.rand(num_of_nodes_prev, num_of_nodes_curr) \
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
        output_without_activation = np.zeros_like(multiply_weights_product)
        output_after_activation = np.zeros_like(multiply_weights_product)
        #multiply_weights_product.shape = (500,150)
        #output.shape = (500,150), for both with and without activation
        #layer_bias.shape = (150, )
        
        #we take out each input from the batch and add the biases to it
        #output_without_activation = z from our notes
        #output_with_activation = a from our notes
        for index, product in enumerate(multiply_weights_product):
                output_without_activation[index] = product+layer_bias
                output_after_activation[index] = self.activation(output_without_activation[index])
        
        #we are returning both because we will need 
        #both these values during back propagation
        return output_without_activation, output_after_activation
    
    def feed_forward(self, input_batch):
        print("Perfoming feed forward")
        self.z = []
        self.a= [input_batch]
        curr_input = input_batch
        for weights, bias in zip(self.weights, self.biases):
            curr_output_without_activation, curr_output_with_activation = \
            self.layer_output(curr_input, weights, bias)
            self.z.append(curr_output_without_activation)
            self.a.append(curr_output_with_activation)
        
        return curr_output_with_activation
        
    def back_propagate(self, final_output, actual_output):
        print("Performing back propagation")
        final_layer_delta = \
        self.cost_function_derivate(final_output, actual_output)*self.activation_derivative(self.z[-1])
        delta_b = [np.zeros(bias.shape) for bias in self.biases]
        delta_w = [np.zeros(weights.shape) for weights in self.weights]
        
        delta_b[-1] = final_layer_delta
        delta_w[-1] = np.dot(np.array([self.a[-2]]).T,final_layer_delta)
        for layer in xrange(2, len(self.structure)):
            delta_layer = np.dot(delta_w[-layer+1], self.weights[-layer+1].T)*self.cost_function_derivate(self.z[-layer])
            delta_b[-layer] = delta_layer
            delta_w[-layer] = np.dot(np.array([self.a[-layer-1]]).T,final_layer_delta)
            
        return delta_b, delta_w
    
    def gradient_descent(self, current_values, deltas):
        return [(current_value - (self.learning_rate * delta)) for current_value, delta in zip(current_values, deltas)]
    
    def update_weights_and_biases(self, delta_biases, delta_weights):
        self.biases = self.gradient_descent(self.biases, delta_biases)
        self.weights = self.gradient_descent(self.weights, delta_weights)
            
    def fit(self, X, Y, learning_rate, epochs, batch_size):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        
        for batch_start_index in xrange(0, len(X), batch_size):
            input_batch = self.X[batch_start_index:(batch_start_index+batch_size)]
            output_batch = self.Y[batch_start_index:(batch_start_index+batch_size)]
            for epoch in xrange(0, epochs):
                print("Starting epoch {0}: ").format(epoch)
                curr_epoch_predicted_output = self.feed_forward(input_batch)
                epoch_delta_b, epoch_delta_w = self.back_propagate(curr_epoch_predicted_output, output_batch)
                self.update_weights_and_biases(delta_biases=epoch_delta_b, delta_weights=epoch_delta_w)
                print("Finished epoch {0}: ").format(epoch)
        
            
                
            
    
    
        