import math
import copy

def sigmoid(s):
    return 1/(1+math.exp(-s))

#Same notation as the one in the backPropagationNN algorithm
def feedForwardNN(units1,units2,w,x):
    NeuralNet = []
    NeuralNet.append(x) #NeuralNet[0]
    NeuralNet.append([1]*units1) #NeuralNet[1]
    NeuralNet.append([1]*units2) #NeuralNet[2]
    NeuralNet.append([1]*2) #NeuralNet[3]
    #Layer 1: Hidden layer 1
    for i in range(units1):
        if(i!= 0):
            s = 0
            for j in range(len(x)):
                s += w[1][j][i] * x[j]
            NeuralNet[1][i] = sigmoid(s)
    #Layer 2: Hidden layer 2
    for i in range(units2):
        if(i!= 0):
            s = 0
            for j in range(units1):
                s += w[2][j][i] * NeuralNet[1][j]
            NeuralNet[2][i] = sigmoid(s)
    #Layer 3: output
    y = 0
    for j in range(units2):
        y += w[3][j][1] * NeuralNet[2][j]
    NeuralNet[3][1] = y

    return NeuralNet


#Back-propagation implementation 2(a) to compute the gradient wrt all weights given one training example
#units_1 and units_2 are the number of units in the hidden layers 1 and 2
#w is a list of lists of lists that returs the corresponding weight given the sollowing notation:
#  w_{12}^1 = w[1][1][2], or, in general, w_{nm}^l = w[l][n][m]
#  w[0] is an empty list
def backPropagationNN(units1,units2,w,example):
    units1 += 1 #This is to count the bias term
    units2 += 1 #This is to count the bias term
    #This is to copy the structure and we will update the values
    L_deriv_w = copy.deepcopy(w)
    #First, we will calculate the values for all the variables given the weights w
    var_NeuralNet = feedForwardNN(units1,units2,w,example[0])
    #This is to copy the structure and we will update the values
    L_deriv_vars = copy.deepcopy(var_NeuralNet)
    
    ##########################################################
    # Layer 3
    ##########################################################
    #Variables
    L_deriv_vars[3][1] = var_NeuralNet[3][1]-example[1]
    #Weights
    for i in range(units2):
        L_deriv_w[3][i][1] = L_deriv_vars[3][1]*var_NeuralNet[2][i]
    
    ##########################################################
    # Layer 2
    ##########################################################
    #Variables
    for i in range(units2):
        if i != 0:
            L_deriv_vars[2][i] = L_deriv_vars[3][1]*w[3][i][1]
    #Weights
    for i in range(units1):
        for j in range(units2):
            if (j != 0):
                L_deriv_w[2][i][j]=L_deriv_vars[2][j]*var_NeuralNet[2][j]*(1-var_NeuralNet[2][j])*var_NeuralNet[1][i]
    
    ##########################################################
    # Layer 1
    ##########################################################
    #Variables
    for i in range(units1):
        L_deriv_vars[1][i] = 0.0
        for j in range(units2):
            if (j != 0):
                L_deriv_vars[1][i] += L_deriv_vars[2][j]*var_NeuralNet[2][j]*(1-var_NeuralNet[2][j])*w[2][i][j]
    #Weights
    for i in range(len(example[0])):
        for j in range(units1):
            if (j != 0):
                L_deriv_w[1][i][j]=L_deriv_vars[1][j]*var_NeuralNet[1][j]*(1-var_NeuralNet[1][j])*var_NeuralNet[0][i]
    return L_deriv_w
