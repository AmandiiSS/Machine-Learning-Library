INSTRUCTIONS TO USE THE PERCEPTRON ALGORITHMS:
------------------------------------------------------

%%%%%%% Standard perceptron algorithm %%%%%%%
standard_perceptron(S,T,r)
where:
1. S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term, and here y is either -1 or 1.
2. The weight vector is a list of [w1 w2 ... wn b]
3. T is the maximum number of epochs
4. r is the learning rate
5. The function returns the final w

%%%%%%% Voted perceptron algorithm %%%%%%%
voted_perceptron(S,T,r)
where:
1. S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term, and here y is either -1 or 1.
2. The weight vector is a list of [w1 w2 ... wn b]
3. T is the maximum number of epochs
4. r is the learning rate
5. The function returns a list of lists of vectors w_m and their votes C_m

%%%%%%% Average perceptron algorithm %%%%%%%
averaged_perceptron(S,T,r)
where:
1. S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term, and here y is either -1 or 1.
2. The weight vector is a list of [w1 w2 ... wn b]
3. T is the maximum number of epochs
4. r is the learning rate
5. The function returns the final a

%%%%%%% Predict functions %%%%%%%
predict_perceptron(w,x)
predict_voted_perceptron(voted_w,x)
predict_averaged_perceptron(a,x)

The imputs are just the results of the corresponding associated functions and an example