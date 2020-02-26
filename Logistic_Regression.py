# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:32:40 2019

@author: willi
"""

import math as m

class Logistic_regression:
    
    def __init__(self, has_momentum, momentum,alpha, learning, variable_count = 3, threshold = 0.5):
        self.learn = learning
        self.weights = [0]*variable_count
        self.limit = threshold
        self.sigma_x = 0
        self.has_momentum = has_momentum
        self.momentum = momentum
        # We need a nesterov speed for each of the weights otherwise 
        # they overwrite each other and dont update correctly
        self.nes_speed = [0]*variable_count
        # Save the L2 regularisation variable value
        self.L2_reg = alpha

    def train_sgd(self, datapoint):
        """
        The datapoint must be received in the form with each variable value
        and the last (chasing) element is the label of the item.
        
        """
        
        if self.has_momentum == True:
            # Update the new weights including Nesterov Momentum
            for i in range(0, len(self.weights)):
                if i == 0:
                    # Using the Nesterov Momentum formula
                    self.nes_speed[i] = self.momentum*self.nes_speed[i] - self.learn*(self.sigma_x-datapoint[-1])*self.sigma_x*(1-self.sigma_x)
                    # The three terms here are the old weight, the current Nesterov speed and the regularisation term
                    self.weights[i] = self.weights[i] + self.nes_speed[i] - self.L2_reg*self.weights[i]
                else:
                    self.nes_speed[i] = self.momentum*self.nes_speed[i] - self.learn*(self.sigma_x-datapoint[-1])*self.sigma_x*(1-self.sigma_x)*datapoint[i-1]
                    self.weights[i] = self.weights[i] + self.nes_speed[i] - self.L2_reg*self.weights[i]
            
        else:
            
            for i in range(0, len(self.weights)):
                # The three terms here are the old weight, the error gradient and the regularisation term
                if i == 0:
                    self.weights[i] = self.weights[i] - self.learn*(self.sigma_x-datapoint[-1])*self.sigma_x*(1-self.sigma_x) - self.L2_reg*self.weights[i]
                else:
                    self.weights[i] = self.weights[i] - self.learn*(self.sigma_x-datapoint[-1])*self.sigma_x*(1-self.sigma_x)*datapoint[i-1] - self.L2_reg*self.weights[i]
    
    def find_sigma(self, datapoint):
        
        t = self.weights[0]
        for i in range(len(datapoint)-1):
            t += self.weights[i+1]*datapoint[i]
        # When the current sigma is found we update the self.sigma_x too (can possibly remove this step in the training method)
        self.sigma_x = 1/(1+m.exp(-t))
        return self.sigma_x
        
    


