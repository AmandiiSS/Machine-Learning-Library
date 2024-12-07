import random
import numpy

#Auxiliar functions
def dot(x, y):
    dot_result = 0
    for i in range(len(x)):
        dot_result += x[i]*y[i]
    return dot_result

#The examples of S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term
#Here y is either -1 or 1
#The weight vector is a list of [w1 w2 ... wn b]

def gauss_kernel(x, y, gamma):
    x_numpy = numpy.array(x)
    y_numpy = numpy.array(y)
    return numpy.exp(-numpy.linalg.norm(x_numpy-y_numpy)**2/gamma)

def kernel_perceptron(S,T,gamma):
    #len(S[0][0]) should be the lenght of [x1 x2 ... xn 1]
    c = [0]*len(S)
    for i in range(T):
        random.shuffle(S)
        for j in range(len(S)):
            y = S[j][1]
            x = S[j][0]
            res = 0
            for k in range(len(x)):
                res += y*c[k]*S[k][1]*gauss_kernel(S[k][0],x,gamma)
            if res <= 0:
                c[j] += 1        
    return c

def predict_perceptron_kernel(c,x,S,gamma):
    res = 0
    for k in range(len(x)):
        res += c[k]*S[k][1]*gauss_kernel(S[k][0],x,gamma)
    if res <= 0:
        return -1
    else:
        return 1
    

S = []
CSVfile = 'bank-note/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_train = line.strip().split(',')
        x = []
        for i in range(len(example_train)-1):
            x.append(float(example_train[i]))
        x.append(1)
        if int(example_train[len(example_train)-1]) == 1:
            y = float(example_train[len(example_train)-1])
        else:
            y = -1
        S.append([x,y])

S_test = []
y_test = []
CSVfile = 'bank-note/test.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_test = line.strip().split(',')
        x = []
        for i in range(len(example_test)-1):
            x.append(float(example_test[i]))
        x.append(1)
        if int(example_test[len(example_test)-1]) == 1:
            y_test.append(1)
        else:
            y_test.append(-1)
        S_test.append(x)

Gammas = [0.1,0.5,1,5,100]
for g in Gammas:
    gamma = g

    c = kernel_perceptron(S,50,gamma)

    total_test_examples = len(S_test)
    total_train_examples = len(S)

    #TRAIN ERROR
    error_train_3b = 0
    i = 0
    for train in S:
        prediction = predict_perceptron_kernel(c,train[0],S,gamma)
        if prediction != train[1]:
            error_train_3b += 1/total_train_examples
        i += 1
    print("Average train prediction error:")
    print(error_train_3b)
    #TEST ERROR
    error_test_3b = 0
    i = 0
    for test in S_test:
        prediction = predict_perceptron_kernel(c,test,S,gamma)
        if prediction != y_test[i]:
            error_test_3b += 1/total_test_examples
        i += 1
    print("Average test prediction error:")
    print(error_test_3b)
