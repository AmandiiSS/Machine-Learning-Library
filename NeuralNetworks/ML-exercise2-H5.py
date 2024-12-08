import neuralnetwork
import random

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

print("Exercise 2a:")
print("--------------")
print("The implementation is done in the neuralnetwork.py file. The implementation has also been debugged (see commented part in the code).")
# #DEBUG
# units1 = 2
# units2 = 2
# #GENERAL (Commented when debugging as well)
# # w = [
# #     [], #Layer 0
# #     [[w_01^1,w_02^1],[w_11^1,w_12^1],[w_21^1,w_22^1]], #Layer 1
# #     [[w_01^2,w_02^2],[w_11^1,w_12^2],[w_21^1,w_22^2]], #Layer 2
# #     [['',w_01^3],['',w_11^3],['',w_21^3]] #Layer 3
# #     ]
# # SPECIFIC EXERCISE 2 and 3 paper problems
# w = [
#     [], #Layer 0
#     [['',-1,1],['',-2,2],['',-3,3]], #Layer 1
#     [['',-1,1],['',-2,2],['',-3,3]], #Layer 2
#     [['',-1],['',2],['',-1.5]] #Layer 3
#     ]
# example = [[1,1,1],1]
# L_deriv_w = neuralnetwork.backPropagationNN(units1,units2,w,example)
# print("Layer 3")
# print(L_deriv_w[3])
# print("Layer 2")
# print(L_deriv_w[2])
# print("Layer 1")
# print(L_deriv_w[1])

print("Exercise 2b:")
print("--------------")
total_test_examples = len(S_test)
total_train_examples = len(S)
T = 100
gamma_0 = 0.001
d = 0.5
gamma = []
for t in range(T):
    gamma.append(gamma_0/(1+(gamma_0/d)*(t+1)))

width = [5,10,25,50,100]
#width = [5]
for wid in width:
    print("Width: ", wid)
    # w0 construction
    w0 = [[]]  # Empty list for input layer (Layer 0)
    w0.append([[random.gauss(0, 1) for _ in range(wid+1)] for _ in range(len(S[0][0]))])  # Weights from input to hidden layer 1
    w0.append([[random.gauss(0, 1) for _ in range(wid+1)] for _ in range(wid+1)])  # Weights from hidden layer 1 to hidden layer 2
    w0.append([[random.gauss(0, 1) for _ in range(2)] for _ in range(wid+1)])  # Weights from hidden layer 2 to output

    result_NN = neuralnetwork.stochastic_gradient_descent_NN(T, w0, S, gamma, wid)

    w = result_NN[0]

    #TRAIN ERROR
    error_train_2b = 0
    i = 0
    for train in S:
        prediction = neuralnetwork.predict_NN(wid,wid,w,train[0])
        if prediction != train[1]:
            error_train_2b += 1/total_train_examples
        i += 1
    print("Average train prediction error:")
    print(error_train_2b)

    #TEST ERROR
    error_test_2b = 0
    i = 0
    for test in S_test:
        prediction = neuralnetwork.predict_NN(wid,wid,w,test)
        if prediction != y_test[i]:
            error_test_2b += 1/total_test_examples
        i += 1
    print("Average test prediction error:")
    print(error_test_2b)

    print("Objective Function values")
    print(result_NN[1])



print("Exercise 2c:")
print("--------------")

total_test_examples = len(S_test)
total_train_examples = len(S)
T = 100
gamma_0 = 0.001
d = 0.5
gamma = []
for t in range(T):
    gamma.append(gamma_0/(1+(gamma_0/d)*(t+1)))

width = [5,10,25,50,100]
#width = [5]
for wid in width:
    print("Width: ", wid)
    # w0 construction
    w0 = [[]]  # Empty list for input layer (Layer 0)
    w0.append([[0 for _ in range(wid+1)] for _ in range(len(S[0][0]))])  # Weights from input to hidden layer 1
    w0.append([[0 for _ in range(wid+1)] for _ in range(wid+1)])  # Weights from hidden layer 1 to hidden layer 2
    w0.append([[0 for _ in range(2)] for _ in range(wid+1)])  # Weights from hidden layer 2 to output

    result_NN = neuralnetwork.stochastic_gradient_descent_NN(T, w0, S, gamma, wid)

    w = result_NN[0]

    #TRAIN ERROR
    error_train_2c = 0
    i = 0
    for train in S:
        prediction = neuralnetwork.predict_NN(wid,wid,w,train[0])
        if prediction != train[1]:
            error_train_2c += 1/total_train_examples
        i += 1
    print("Average train prediction error:")
    print(error_train_2c)

    #TEST ERROR
    error_test_2c = 0
    i = 0
    for test in S_test:
        prediction = neuralnetwork.predict_NN(wid,wid,w,test)
        if prediction != y_test[i]:
            error_test_2c += 1/total_test_examples
        i += 1
    print("Average test prediction error:")
    print(error_test_2c)

    print("Objective Function values")
    print(result_NN[1])