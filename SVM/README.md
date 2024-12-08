INSTRUCTIONS TO USE THE SVM ALGORITHMS:
------------------------------------------------------

%%%%%%% Primal Stochastic Subgradient Descent Svd algorithm %%%%%%%
primal_stochastic_subgradient_descent_svm(S,T,gamma,C)
where:
1. S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term, and here y is either -1 or 1.
3. T is the maximum number of epochs
4. gamma is the learning rate
5. C is the slack trade variable
6. The function returns the final w


%%%%%%% Predict functions %%%%%%%
predict_svm(w,x)

The input is just the result of the corresponding svm function and an example