import neuralnetwork

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