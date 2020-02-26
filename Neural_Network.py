

import Logistic_Regression as LR
import copy
import random as r
import numpy as np


"""
The main principle of design of this code is that the neural network (NN) is a list of nodes, each of which
is a logisitc regression (as is the most elementary NN design).


"""

class NeuralNet:
    """Minimal handling of a hidden layer of size zero included as it wouldnt make sense to insert"""    
    def __init__(self, dataset, learn_rate = 0.001, hidden_layers = (5, 2), output_layer = 1, threshold = 0.5, alpha = 0.0001, momentum = 0.9, shuffle = True, Epochs = 200):
        #Set the learning rate
        self.learn_rate = learn_rate
        # Alpha is the Regularisation term
        self.alpha = alpha
        # Whether to shuffle the training data on each epoch
        self.shuffle = shuffle
        # Set the treshold for binary classification
        self.threshold = threshold
        # If momentum is non-zero make sure a momentum term is included
        if momentum > 0:
            self.has_momentum = True
            self.momentum = momentum
        else: 
            self.has_momentum = False
            self.momentum = momentum
        # Epochs refer to how many times to iterate over the training data in its entireity 
        self.epochs = Epochs
        
        
        # Create the layers and populate each layer with nodes for the network
        all_layers = []
        k = 0
        # Generate each layer to be filled with the correct number of "nodes" as specified by the user
        for i in range(0, len(hidden_layers)):
            # Only allow non-zero layers to be created
            if hidden_layers[i] != 0:
                if len(all_layers) == 0:
                    all_layers.append([])
                    if i < len(hidden_layers):
                        #print(hidden_layers[i])
                        # initialise a standard node for the layer with random initial weights
                        # Only the length of the dataset is required as the length includes the label 
                        # even though the label isn't used (saves adding + 1 for the padding weight)
                        layer_node = LR.Logistic_regression(learning = self.learn_rate, alpha = self.alpha, variable_count=(int(len(dataset[0]))), threshold=threshold, has_momentum = self.has_momentum, momentum = self.momentum)
                        for node_count in range(hidden_layers[i]):
                            for weight in range(len(layer_node.weights)):
                                # The random weights are such that
                                layer_node.weights[weight] = r.normalvariate(0,1)
                            # Deep copy required to that the sub-details are not identical
                            all_layers[k].append(copy.deepcopy(layer_node))
                    
                        k += 1
                else: 
                    all_layers.append([])
                    if i < len(hidden_layers):
                        #print(hidden_layers[i])
                        # initialise a standard node for the layer with random initial weights
                        layer_node = LR.Logistic_regression(learning=self.learn_rate,alpha = self.alpha, variable_count=(int(hidden_layers[i-1])+1), threshold=threshold, has_momentum=self.has_momentum, momentum = self.momentum)
                        for node_count in range(hidden_layers[i]):
                            for weight in range(len(layer_node.weights)):
                                # The random weights are such that
                                layer_node.weights[weight] = r.normalvariate(0,1)
                            # Deep copy required to that the sub-details are not identical
                            all_layers[k].append(copy.deepcopy(layer_node))
                    
                    k += 1
                    # K is used as a substitute for i to account for the zeros in tuples
                    # It is a quick fix and will be removed if time
                    
            # The final lines capture the output layer and makes it of a set size
            # Most cases do not require multiple outputs, but this code was designed
            # so that it could be extended to a CNN       
        all_layers.append([])
        layer_node = LR.Logistic_regression(learning=self.learn_rate,alpha = self.alpha, variable_count=(int(hidden_layers[-1])+1), threshold=threshold, has_momentum=self.has_momentum, momentum = self.momentum)
        for node_count in range(output_layer):
            for weight in range(len(layer_node.weights)):
                layer_node.weights[weight] = r.normalvariate(0,1)
            all_layers[k].append(copy.deepcopy(layer_node))
        self.layers = all_layers

    
    # The fitting function takes the training data and trains the network by updating the weights
    def fit(self, datafile):
        epoch = 0
        while epoch < self.epochs:
            # If you want to shuffle the training data each epoch this must occur each time
            if self.shuffle == True:
                data = copy.deepcopy(datafile)
                np.random.shuffle(data)
            else:
                data = copy.deepcopy(datafile)
            
            # Train over all rows in the data
            for row in range(0, len(data)):
                #print(self.layers[0][0].weights)
                """The forward pass through"""
                # Create a place holder list to update the sigma values
                hold_sigma = []
                
                # Update the sigma values a layer at a time starting at the first hidden layer
                for layer in range(len(self.layers)):
                    # Append a new row to the sigma list
                    hold_sigma.append([])
                    # If the first layer use the data to fit the sigma values
                    if layer == 0:
                        for i in range(len(self.layers[layer])):
                            hold_sigma[layer].append(self.layers[layer][i].find_sigma(data[row]))
                        # Make the list element of each following layer the label of the data row
                        # This is done for the original data set and the updated set 
                        hold_sigma[layer].append(data[row][-1])
                    else:
                        for i in range(len(self.layers[layer])):
                            hold_sigma[layer].append(self.layers[layer][i].find_sigma(hold_sigma[layer-1]))     
                        hold_sigma[layer].append(data[row][-1])
                """The backward pass through"""
                # The backwards pass loop is separate from the forwards pass loop as it has to act in reverse
                # We start at 1 so the second for loop works correctly
                for layer in range(1,len(self.layers)+1):
                    # We want to iterate backwards through each layer so we do the last layer first
                    if layer != len(self.layers):
                        for i in range(len(self.layers[-layer])):
                            self.layers[-layer][i].train_sgd(hold_sigma[-layer-1])
                    else:
                        for i in range(len(self.layers[-layer])):
                            self.layers[-layer][i].train_sgd(data[row])
            epoch += 1
        
    def predict(self, test_data):
        # The prediction is simply done by using a forward pass through the network and taking the output
        # Create a place holder list to update the sigma values
        hold_sigma = []
        
        # Update the sigma values a layer at a time starting at the first hidden layer
        for layer in range(len(self.layers)):
            # Append a new row to the sigma list
            hold_sigma.append([])
            # If the first layer use the data to fit the sigma values
            if layer == 0:
                for i in range(len(self.layers[layer])):
                    hold_sigma[layer].append(self.layers[layer][i].find_sigma(test_data))
                # Make the list element of each following layer the label of the data row
                # This is done for the original data set and the updated set 

            else:
                for i in range(len(self.layers[layer])):
                    hold_sigma[layer].append(self.layers[layer][i].find_sigma(hold_sigma[layer-1]))     
        
        if hold_sigma[-1][-1] < self.threshold:
            prediction = 0
        elif hold_sigma[-1][-1] > self.threshold:
            prediction = 1
        #print(hold_sigma[-1][-1])
        return prediction
