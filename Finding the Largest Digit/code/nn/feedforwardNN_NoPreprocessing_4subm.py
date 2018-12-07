# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 14:00:02 2018

@author: yfoucault002
"""

# =============================================================================
# Feedforward Neural Network (Question 2)
# =============================================================================
# 
# Architecture 1 : [4096, 200, 10] 
#  so only 1 hidden layer, and 200 nodes in that layer
# Note that the output layer has 10 nodes, one for each possible digit (one-hot encoding)
# =============================================================================



#### Libraries
# Standard library
#import random

# Third-party libraries
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# =============================================================================
# Load data
# =============================================================================
X_raw = np.loadtxt("train_x.csv", delimiter=",") # load from text 
Y_raw = np.loadtxt("train_y.csv", delimiter=",") 
#X = X.reshape(-1, 64, 64) # reshape 
#Y = Y.reshape(-1, 1) 
##scipy.misc.imshow(x[0]) # to visualize only  # RuntimeError: Could not execute image viewer.




# =============================================================================
# Define useful functions
# =============================================================================


def actiFun(z):   # activation function
    return np.divide(1.0 , (1.0 + np.exp(-z)) )   # defined as the sigmoid function


def actiPrime(z):
    return actiFun(z) * ( 1 - actiFun(z) )   # first derivative of the activation function (here, sigmoid function)


def costPrime(a, y):
    return (a - y)     # derivative of cost function, here MSE # Beware the sign

#def costFun(batchSize, a, y):
#    return  0.5 * np.square(y - a)      # never used actually

# =============================================================================
# Input manip
# =============================================================================



nF = 4096 # number of features by image



X = [(u/255)*0.0001 for u in X_raw]   # rescale; and divide by 100000; otherwise, the first matrix multiplication
# of the weights w_H by the x features returns a zH in the 4 digits, and the exp(-zH) = 0 or close, 
#and every aH = 1, so gradient will vanish. Simply because there are so many (4096) features.
# and then again one more increase by a multiple of 10 (function of number of nodes in hidden layer)



X = [np.reshape(u, (nF, 1)) for u in X]    # add 1 dimension to each array in list


nX = len(X)


# reshape feature vector from 64 x 64 to 1 X 4096
#X = np.reshape(X,(nX, nF))



#Y_raw = Y_raw.reshape(-1,1)
Y_raw = [np.reshape(u, (1,1)) for u in Y_raw]   # each array in list gets 1 more dimension; tip: this works better than code line above to ensure shape is subsequently preserved


#Y = np.zeros((nX, 10))     
#y = y.reshape(-1,1)   # 10-by-1

# Train vs Validation set split 
X_train, X_valid, Y_trainRaw, Y_valid = train_test_split(X, Y_raw, test_size=0.2, random_state=42)   




dataValid = zip(X_valid,Y_valid)   # Y_test remains as such (meaning not converted to a one-hot vector; just the digit)


nValid = len(dataValid)


#reshaping Y_train:
nTrn = len(Y_trainRaw)

for n in range(nTrn):
    Y_trainRaw[n] = Y_trainRaw[n].astype(int)

Y_train = np.zeros((nTrn,10)) # initialize y as one-hot vector

for n in range(nTrn):
    Y_train[n, Y_trainRaw[n]] = 1    # converts train Examples to one-hot vector


Y_train = [np.reshape(u,(10,1)) for u in Y_train]    # convert array to list of 10-by-1 arrays
#
#
#Y_train[:, Y_trainRaw] = 1

dataTrain = zip(X_train, Y_train)    # this line takes several seconds to run




# =============================================================================
# Hyperparameters
# =============================================================================

modelNbr = 2 #model number 

batchSize = 30

alpha = 10 #  0.1      # learning rate : 0.1 too low; saturates because of that?

nLayers = 1   # number of hidden layers 

nH = 30 # 200 # number of nodes in hidden layer

nL = 10 # number of nodes in output layer (as many as classes)

nEpochs = 100 # number of learning steps

iniParamDistri = 1    # 0 # 0 for uniform distribution, 1 for standard Gaussian distribution



# =============================================================================
# Train the network
# =============================================================================

accuRates = []  # initialize list of accuracy rates (1 for each epoch)
accuRate = 0.0 # initialize accuracy rate to long



# =============================================================================
# Initialize biases and weights
# =============================================================================

if iniParamDistri == 0:
    
    bH = np.random.rand(nH, 1)     
    
    bL = np.random.rand(nL, 1) 
    
    wH = np.random.rand(nH, nF)  # initialize bias term vector for hidden layer (uniform distribution)
    
    wL = np.random.rand(nL, nH)  

else:
    
    bH = np.random.randn(nH, 1)     
    
    bL = np.random.randn(nL, 1) 
    
    wH = np.random.randn(nH, nF)  # initialize bias term vector for hidden layer (standard Gaussian distribution)
    
    wL = np.random.randn(nL, nH)  


# note for report: At first, we initialized weights from uniform distribution between 0 and 1; as a result, all predictions were set to 1 (meaning proba of 1 for every of the 10 digits; not what we want)

# initialize gradient (to ensure they are long format, not integer!!!! otherwise get rounded down to zero!)

nabla_wH = np.zeros((nH, nF))   # default dtype = float

nabla_wL = np.zeros((nL, nH))

nabla_bH = np.zeros((nH, 1))

nabla_bL = np.zeros((nL, 1))




for n in range(nEpochs):
    
    k = np.random.choice(nTrn - batchSize + 1, 1) # starting index of minibatch (cannot be more than nX minus batchSize + 1)
    

      
    nablas_wL = [] # list of gradients of weights associated with connections to output layer for the various examples in mini-batch
    nablas_bL = []
    nablas_wH = []
    nablas_bH = []
    
    for s in range(batchSize):
   
        sample = dataTrain[ k[0] + s ]    # recall that s starts at 0, so we start at k[0] and end at the end of the batch, i.e. k[0] + batchSize - 1 index
        
        x = sample[0]
        
        #x = x.reshape(-1,1)    # 4096-by-1
                
                
        y = sample[1]
        
#                 
        # =============================================================================
        # Stochastic Gradient descent
        # =============================================================================
        # 
        
        # =============================================================================
        # Forward pass
        # =============================================================================
        
        zH = np.dot( wH, x ) + bH          # calculate weighted input vector
        
        aH = actiFun(zH)      # calculate activation for hidden nodes
        
        zL = np.dot( wL, aH ) + bL
            
        aL = actiFun(zL)      # calculate activation for output nodes (which is also the prediction)
        #RETURNS A 10X1 VECTOR OF ONES!!!!!! zL IS TOO LARGE, AT 200 OR SO
       # =============================================================================
        # Backpropagation
        # =============================================================================
        
        # Output layer:
        #deltaL = costPrime(aL, y) * aL * (1 - aL)  
        deltaL = costPrime(aL, y) * actiPrime(aL)
        
        nabla_bL = deltaL   # gradient for bias term associated with output node
        
        nabla_wL = np.dot(deltaL, aH.T) # gradient with respect to weights linking hidden layer to output layer; nL-by-nH (matches wL shape, with as many rows as output nodes, and as many columns as input (here Hidden layer) nodes)
        
        # Hidden layer:
        deltaH =  np.dot(wL.T, deltaL) # nH-by-nL matmult with nL-by-1 => nH-by-1 
        deltaH = deltaH * actiPrime(aH)    # elementwise multiplication
        
        nabla_bH = deltaH    # hidden layer bias gradient
          
        nabla_wH = np.dot(deltaH, x.T)  # hidden layer weight gradients
        
        # Stores gradients for weights and biases in list
        nablas_wL.append(nabla_wL)
        nablas_bL.append(nabla_bL)
        nablas_wH.append(nabla_wH)
        nablas_bH.append(nabla_bH)
        
        
    # Calculate averages over the mini-batch
    
    # output layer biases average gradients:
    tempArray = np.zeros((nL,1))   #same size as nabla_bL
    avg = 0.0
        
    
    for a in range(batchSize):
        tempArray += nablas_bL[a]  
    avg = tempArray / batchSize
    meanNabla_bL = avg
    tempArray = np.zeros((nL,1)) # reinitialize gradient array
    
    # hidden layer biases average gradients:
    tempArray = np.zeros((nH,1))
    
    for a in range(batchSize):
        tempArray += nablas_bH[a]  
    avg = tempArray / batchSize
    meanNabla_bH = avg
    tempArray = np.zeros((nH,1)) # reinitialize gradient array
    
      
    
    
    # output layer weights average gradients:
#    tempVal = 0.0
#    avg = 0.0
    
    tempArray = np.zeros((nL, nH))
#    
    for a in range(batchSize):
        tempArray += nablas_wL[a]  
    avg = tempArray / batchSize
    meanNabla_wL = avg
    tempArray = np.zeros((nL, nH)) # reinitialize gradient array
      
     
    # hidden layer weight average gradients:
     
    tempArray = np.zeros((nH, nF))
#    
    for a in range(batchSize):
        tempArray += nablas_wH[a]  
    avg = tempArray / batchSize
    meanNabla_wH = avg
    tempArray = np.zeros((nH, nF)) # reinitialize gradient array
     
   
    
    # Update biases and weights for the sample being processed
    
    bL = bL - alpha * meanNabla_bL
    bH = bH - alpha * meanNabla_bH
    
    wL = wL - alpha * meanNabla_wL
    wH = wH - alpha * meanNabla_wH  
    
    
        
    # =============================================================================
    # Cross-validation - Make prediction and evaluate prediction accuracy
    # =============================================================================
        
    nAccuPred = 0.0    # counter to store number of accurate predictions  
        
    #for j in [v for v in range(nX) if v not in range (k, k+batchSize)]:   #looping through all examples, except examples selected as training examples
    for j in range(nValid): 
        # Manipulate data
        sample = dataValid[j]
    
        x = sample[0]    # 1-D array
        #x = x.reshape(-1, 1)   # add one axis
    
        
        
        yValid = sample[1]
    
        
#        for n in range(10):
#            yRaw[n] = int(yRaw[n])
#        
#        y = np.zeros(10)     # reset y to one-hot vector
#        y = y.reshape(-1,1)   # 10-by-1
#
#                        
#        y[yRaw] = 1  # converts sample to one-hot vector
#    
        # Forward pass
        zH = np.dot( wH, x ) + bH          # calculate weighted input vector
    
        aH = actiFun(zH)      # calculate activation for hidden nodes
       
        zL = np.dot( wL, aH ) + bL
        
        aL = actiFun(zL)      # calculate activation for output nodes (which is also the prediction)
    
        # Make prediction
        digitPred = np.argmax(aL)     #Returns the index of the element with the highest value in the 10-size one-hot vector
    
#        yRaw = np.nonzero(y)
#        yRaw = yRaw[0]
#        yRaw = yRaw[0]
        
        predVal = (digitPred == yValid) #Returns 1 if prediction is accurate, 0 otherwise
    
        nAccuPred = nAccuPred + predVal
        

    #Print accuracy attained at the end of epoch n
    
    print "Epoch {0}: {1} / {2}".format(n, nAccuPred, nValid)  
    
    
    accuRate = nAccuPred / ( nValid )
    accuRates.append(accuRate)
    
## =============================================================================
## Plot learning curve (accuracy rates vs. epoch index)
## =============================================================================
    
epochList = range(1, nEpochs+1)         #index epochs from 1 to number of epochs 



accuRatesVal = np.zeros((nEpochs))
    
for v in range(nEpochs):
    accuRatesVal[v] = accuRates[v]     #convert list to array

   
plt.figure(modelNbr)
    #plt.subplot(212)
    #plt.title('Validation data set fit')
plt.title('Feedforward Neural Network Model %d Learning Curve' %modelNbr)
plt.xlabel('Epochs')
plt.ylabel('Accuracy rate')
plt.plot( epochList , accuRatesVal ,  'k-')

plt.show()   
    
    
    
    
    
    
    
    
    
     
    
    
    
    
    
    
    
    
    
    
     
    