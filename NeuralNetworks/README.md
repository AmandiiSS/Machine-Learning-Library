INSTRUCTIONS TO USE THE NN ALGORITHMS:
------------------------------------------------------

%%%%%%% Backpropagation and feedforward NN algorithm %%%%%%%
feedForwardNN(units1,units2,w,x)
backPropagationNN(units1,units2,w,example)
where:
1. units_1 and units_2 are the number of units in the hidden layers 1 and 2 (without considering the bias term)
2. w is a list of lists of lists that returs the corresponding weight given the following notation:
w_{12}^1 = w[1][1][2], or, in general, w_{nm}^l = w[l][n][m]
w[0] is an empty list
3. example is an example [x,y]
4. x is the part of an example corresponding to the input variables
5. The feedforward function returns all the values of the nodes in all layers given a specific x
6. The backpropagation function returns all the derivatives of L (the loss function) wrt each w


%%%%%%% Stochastic Gradient Descent NN algorithm %%%%%%%

stochastic_gradient_descent_NN(T, w0, S, gamma, width)
1. S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term, and here y is either -1 or 1.
3. T is the maximum number of epochs
4. gamma is the learning rate vector created as asked in the statement
5. w0 is the w "matrix" described before but being an initial guess
6. The function returns the final w matrix and the objective function values corresponding to each epoch


%%%%%%% Predict function %%%%%%%

predict_NN(units1, units2, w,x)
The imput is just the result of the corresponding NN function and an example, and the units1 and units2 are the number of units in the layers 1 and 2 (without considering the bias term)