import time
import Logistic_Regression as LR
import Neural_Network as NN
import random as r
import numpy as np # Numpy is only used for the oversampling method

## This file uses a hand-built k-fold validation mechanism to train and test the neural network I created with a range of 
# hidden layers and number of nodes per layer



""" ----------------- IMPORTING DATA ------------- """

base_data =[]
# import the climate simulation data set
# The file here has been modified sot hat the first row removed
# for ease of import and was converted to csv for easier interpretation
# by the following code

raw_clim = open("climate_data.csv")

while True:
    # Set the current line to be the next unseen line of the pulsar data
    current_line = raw_clim.readline()
    if len(current_line) == 0:
        break
    current_data = current_line.split(",")
    for item in range(len(current_data)):
        # Use rstrip as the \n escape command is otherwise imported
        current_data[item] = str(current_data[item]).rstrip()
    base_data.append(current_data)

raw_clim.close()

# Removal of the first and second rows
for i in range(len(base_data)):
    base_data[i].pop(0)
    base_data[i].pop(0)
    for j in range(len(base_data[i])):
        base_data[i][j] = float(base_data[i][j])
    
"""----------Check for class imbalance----------"""
# This confirms that there are 46 failures out of the 540 data items.
fail_count = 0
success_count = 0
for i in range(0, len(base_data)):
    if base_data[i][18] == 0:
        fail_count += 1
    elif base_data[i][18] == 1:
        success_count += 1
        
print(success_count/fail_count)


"""-------- Use my oversampling code that was created in the datamining project to balance the data -------"""

# Define my oversampling method
def oversample(training_data, classifications):
    oversampled = []
    oversample_classification = []
    
    # Count each positive and negative result in the training set to oberve bias
    class1 = 0 # Negative class
    class2 = 0 # Positive class
    for item in range(len(training_data)):
       oversampled.append(training_data[item])
       oversample_classification.append(classifications[item])
       if classifications[item] == 0:
           class1 += 1
       elif classifications[item] == 1:
           class2 += 1
    # Find the ratio of negative to positive 
    ratio = class1/class2
    
    # Minority Class 
    minority_class = None
    # Set it to be a 1:1.2 ratio minimum
    if ratio < 0.833:
        minority_class = 0
    elif ratio > 1.2:
        minority_class = 1
    
    if minority_class is not None:
        # Create a list of the minority class
        minority_set = []
        minority_classification = []
        for i in range(0,len(training_data)):
            # If the classification is in the minority class add this to the minority data set for sampling
            if classifications[i] == minority_class:
                minority_set.append(training_data[i])
                minority_classification.append(classifications[i])
        
        if ratio > 1:
            while ratio > 1.2:
                # set the random seed to change so the same sample isnt taken each time
                r.seed(ratio)
                # Find a minority set element index to append
                new_index = r.sample(range(len(minority_set)), 1)[0]
                # append the re-sampled information to the oversampled set
                oversampled.append(minority_set[new_index])
                # Append the re-sampled classification to the classification set
                oversample_classification.append(minority_classification[new_index])
                # Count each positive and negative result in the training set to oberve bias
                class1 = 0 # Negative class
                class2 = 0 # Positive class
                for item in range(0,len(oversampled)):
                    if oversample_classification[item] == 0:
                        class1 += 1
                    elif oversample_classification[item] == 1:
                        class2 += 1
                ratio = class1/class2
            return np.array(oversampled), np.array(oversample_classification)
        
        elif ratio < 1:
            while ratio < 0.833:
                # set the random seed to change so the same sample isnt taken each time
                r.seed(ratio)
                # Find a minority set element index to append
                new_index = r.sample(range(len(minority_set)), 1)[0]
                # append the re-sampled information to the oversampled set
                oversampled.append(minority_set[new_index])
                # Append the re-sampled classification to the classification set
                oversample_classification.append(minority_classification[new_index])
                # Count each positive and negative result in the training set to oberve bias
                class1 = 0 # Negative class
                class2 = 0 # Positive class
                for item in range(0,len(oversampled)):
                    if oversample_classification[item] == 0:
                        class1 += 1
                    elif oversample_classification[item] == 1:
                        class2 += 1
                ratio = class1/class2
            return oversampled, oversample_classification
        
    # If there is mininmal or no class imbalance return the original data sets
    else:
        return training_data, classifications

"""------- Using my cross-evaluation code from the data mining project ---------"""
# Arbitrarily used split generator so it is consistent
r.seed(100)
shuffle_indices = r.sample(range(len(base_data)), int(len(base_data)))

shuffled_data =  []

for i in shuffle_indices:
    shuffled_data.append(base_data[i])
cross_val_splits = []
shift_amount = 1/5
# 2 minimums and 2 maximums for each data set if there is a partition.
min_traininga = 0
min_trainingb = 0
max_trainingb = 0
min_testa = 0 # Create the lower test set values
max_testa = 0

# First add the first split for the basic 90% train, 10% test split
    
cross_val_splits.append([[int(0*len(shuffled_data)), int(0.8*len(shuffled_data))], [0, 0],
                          [int(0.8*len(shuffled_data)),int(1*len(shuffled_data))]])
# The while loop finds all possible splits in a rotating fashion
while min_traininga < 0.8:
    min_traininga += shift_amount
    # The lower bound of the test set has to move if the upper bound has already moved
    if max_testa > 0:
        min_testa += shift_amount
    max_testa += shift_amount
    
    if min_testa > 0:
        max_trainingb += shift_amount
    
    
    min_trainingb = round(min_trainingb, 1)
    max_trainingb = round(max_trainingb, 1)
    min_testa = round(min_testa, 1)
    max_testa = round(max_testa, 1)
    min_traininga = round(min_traininga, 1)  
    # The upper bound of each set is fixed 
    cross_val_splits.append([[int(min_traininga*len(shuffled_data)), int(1*len(shuffled_data))], 
                     [int(min_trainingb*len(shuffled_data)), int(max_trainingb*len(shuffled_data))],[int(min_testa*len(shuffled_data)),int(max_testa*len(shuffled_data))]])


file = open("Climate_NN_SGD_OVERSAMPLED_Performance_Dataaa.txt", "w")
file.write("The format of this data is cross validation number,the number of layers, the number of nodes per layer,")
file.write("the true positives, true negatives, false positives, false negatives, network creation time, network fit time, network predict time\n")
   
    
# Create empty test and training sets each step of cross validation


for i in range(4,5):#len(cross_val_splits):
        training_set = []
        test_set = []
        # Create the training set partitions
        classified_data = []
        # Record each partition made
        partition1 = shuffled_data[cross_val_splits[i][0][0]:cross_val_splits[i][0][1]]
        partition2 = shuffled_data[cross_val_splits[i][1][0]:cross_val_splits[i][1][1]]
        partition3 = shuffled_data[cross_val_splits[i][2][0]:cross_val_splits[i][2][1]]
        
        # Only fill the training set with a partition if the partition isn't empty
        # I fill one element at a time as I am using regular lists and cannot use numpy vstack
        for j in range(len(partition1)):
            if len(partition1) > 0:
                training_set.append(partition1[j])
        for j in range(len(partition2)):
            if len(partition2) > 0:
                training_set.append(partition2[j])
        for j in range(len(partition3)):
            test_set.append(partition3[j])
        

        X  = []
        numeric_variables = []
        y = []
        for l in range(len(training_set)):
            X.append(training_set[l][0:-2])
            y.append(training_set[l][-1])

        X_train, y_train = oversample(X, y)
        X_train = np.c_[X_train, y_train]
        # Create a neural network using my implementation
        # solver refers to the method of solving, in this instance it is 
        # stochastic gradient descent
        # Hidden layer sizes refers to the number of layers (length of tuple) and the number of nodes
        # (number at each point in tuple)
        """----- Test the neural network over different numbers of layers and difference depths ------"""
        # testing up to 4 layers each layer with up to 10 nodes
        l1,l2,l3,l4 = 7,8,0,0
        layer_count = 4

        while l1 < 15:
            if l2 < 15:
                l2 += 1
            elif l2 == 15:
               l1 += 1
               l2 = 8
            
            # Record 3 times taken, the time to generate the network, the time to fit the network
            # and the time to predict using the network
            generate_start_time = time.perf_counter()
            climate_NN = NN.NeuralNet(dataset=X_train,learn_rate=0.001, hidden_layers=(l4, l3, l2, l1), alpha = 0.0001)
            generate_total = time.perf_counter() - generate_start_time
            fit_start_time = time.perf_counter()
            climate_NN.fit(X_train)
            fit_total = time.perf_counter() - fit_start_time
            # Set up the classification
            X_test  = []
            y_test = []
            X_test_numeric_variables=[]
            for k in range(len(test_set)):
                X_test.append(test_set[k][0:-1])
                y_test.append(test_set[k][-1])
            
            predict_start_time = time.perf_counter()
            predictions = []
            for point in X_test:
                predictions.append(climate_NN.predict(point))
            predict_total = time.perf_counter()-predict_start_time
            
            classified_data = []
            for q in range(len(predictions)):
                classified_data.append([y_test[q], predictions[q]])
        
            # Find the accuracy, precision and recall metrics for the current partition and depth
            tpr,tnr,fpr,fnr = 0,0,0,0
            for k in range(len(classified_data)):
                if classified_data[k][0] == 0 and classified_data[k][1] == 0:
                    tnr += 1
                elif classified_data[k][0] == 1 and classified_data[k][1] == 1:
                    tpr += 1
                elif classified_data[k][0] == 0 and classified_data[k][1] == 1:
                    fpr += 1
                elif classified_data[k][0] == 1 and classified_data[k][1] == 0:
                    fnr += 1
                    
            # Store all items to be written to file in a list
            items = [i,layer_count,  l1,l2,l3,l4,tpr,tnr,fpr,fnr,generate_total,fit_total, predict_total]
            for info in range(len(items)):
                if info == len(items)-1:
                    file.write(" " + str(items[info]) + " \n")
                elif info % 2 == 0:
                    file.write("  " + str(items[info]))
                else:
                    file.write(", " + str(items[info]) + " ,".rstrip())
            
            
file.close()











    
