INSTRUCTIONS TO USE THE LINEAR REGRESSION ALGORITHMS:
------------------------------------------------------

%%%%%%% Gradient descent algorithm %%%%%%%
GradientDescent (w0,X,y,r,max_it)
where:
1. w0 is the initial guess for w
2. X is a list of list (each of them being [1 x_1 ... x_n] and there are m of them)
3. y is a list [y_1 ... y_m]
4. r is the learning rate
5. max_it is the maximum number of iterations that we let the algorithm to run
The function returns [w,converge, it, function_values]
where:
1. w is the final guess
2. converge is a boolean that says if the algorithm has converged or not
3. it is the number of iterations that the algorithm has done before either converging or reaching the maximum
4. function_values is a list with the value of the cost function at each iteration

%%%%%%% Stochastic Gradient descent algorithm %%%%%%%
StochGradientDescent (w0,X,y,r,max_it)
where:
1. w0 is the initial guess for w
2. X is a list of list (each of them being [1 x_1 ... x_n] and there are m of them)
3. y is a list [y_1 ... y_m]
4. r is the learning rate
5. max_it is the maximum number of iterations that we let the algorithm to run
The function returns [w,converge, it, function_values]
where:
1. w is the final guess
2. converge is a boolean that says if the algorithm has converged or not
3. it is the number of iterations that the algorithm has done before either converging or reaching the maximum
4. function_values is a list with the value of the cost function at each iteration