import svm

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


print("Exercise 2a and 2c:")
print("---------------------")
#Tune gamma_0 > 0 and a > 0 to ensure converge of the algorithm
gamma_0 = 0.00001
a = 0.5
#Building of the gamma vector
gamma = [0]*100
for i in range(100):
    gamma[i] = gamma_0/(1+(gamma_0/a)*i)
T = 100
total_test_examples = len(S_test)
total_train_examples = len(S)

print("C = 100/873")
print("**************")
C1 = 100/873
w_obj_func_2a_C1 = svm.primal_stochastic_subgradient_descent_svm(S,T,gamma,C1)
print(w_obj_func_2a_C1[1])
w_2a_C1 = w_obj_func_2a_C1[0]
#TRAIN ERROR
error_train_2a_C1 = 0
i = 0
for train in S:
    prediction = svm.predict_svm(w_2a_C1,train[0])
    if prediction != train[1]:
        error_train_2a_C1 += 1/total_train_examples
    i += 1
print("Average train prediction error:")
print(error_train_2a_C1)
#TEST ERROR
error_test_2a_C1 = 0
i = 0
for test in S_test:
    prediction = svm.predict_svm(w_2a_C1,test)
    if prediction != y_test[i]:
        error_test_2a_C1 += 1/total_test_examples
    i += 1
print("Average test prediction error:")
print(error_test_2a_C1)
print("w found:")
print(w_2a_C1)

print("C = 500/873")
print("**************")
C2 = 500/873
w_obj_func_2a_C2 = svm.primal_stochastic_subgradient_descent_svm(S,T,gamma,C2)
print(w_obj_func_2a_C2[1])
w_2a_C2 = w_obj_func_2a_C2[0]
#TRAIN ERROR
error_train_2a_C2 = 0
i = 0
for train in S:
    prediction = svm.predict_svm(w_2a_C2,train[0])
    if prediction != train[1]:
        error_train_2a_C2 += 1/total_train_examples
    i += 1
print("Average train prediction error:")
print(error_train_2a_C2)
#TEST ERROR
error_test_2a_C2 = 0
i = 0
for test in S_test:
    prediction = svm.predict_svm(w_2a_C2,test)
    if prediction != y_test[i]:
        error_test_2a_C2 += 1/total_test_examples
    i += 1
print("Average test prediction error:")
print(error_test_2a_C2)
print("w found:")
print(w_2a_C2)

print("C = 700/873")
print("**************")
C3 = 700/873
w_obj_func_2a_C3 = svm.primal_stochastic_subgradient_descent_svm(S,T,gamma,C3)
print(w_obj_func_2a_C3[1])
w_2a_C3 = w_obj_func_2a_C3[0]
#TRAIN ERROR
error_train_2a_C3 = 0
i = 0
for train in S:
    prediction = svm.predict_svm(w_2a_C3,train[0])
    if prediction != train[1]:
        error_train_2a_C3 += 1/total_train_examples
    i += 1
print("Average train prediction error:")
print(error_train_2a_C3)
#TEST ERROR
error_test_2a_C3 = 0
i = 0
for test in S_test:
    prediction = svm.predict_svm(w_2a_C3,test)
    if prediction != y_test[i]:
        error_test_2a_C3 += 1/total_test_examples
    i += 1
print("Average test prediction error:")
print(error_test_2a_C3)
print("w found:")
print(w_2a_C3)







print("Exercise 2b:")
print("-------------")
#Gamma_0 from the previous exercise
gamma_0 = 0.00001
#Building of the gamma vector
gamma = [0]*100
for i in range(100):
    gamma[i] = gamma_0/(1+i)
T = 100
total_test_examples = len(S_test)
total_train_examples = len(S)

print("C = 100/873")
print("**************")
C1 = 100/873
w_obj_func_2b_C1 = svm.primal_stochastic_subgradient_descent_svm(S,T,gamma,C1)
print(w_obj_func_2b_C1[1])
w_2b_C1 = w_obj_func_2b_C1[0]
#TRAIN ERROR
error_train_2b_C1 = 0
i = 0
for train in S:
    prediction = svm.predict_svm(w_2b_C1,train[0])
    if prediction != train[1]:
        error_train_2b_C1 += 1/total_train_examples
    i += 1
print("Average train prediction error:")
print(error_train_2b_C1)
#TEST ERROR
error_test_2b_C1 = 0
i = 0
for test in S_test:
    prediction = svm.predict_svm(w_2b_C1,test)
    if prediction != y_test[i]:
        error_test_2b_C1 += 1/total_test_examples
    i += 1
print("Average test prediction error:")
print(error_test_2b_C1)
print("w found:")
print(w_2b_C1)

print("C = 500/873")
print("**************")
C2 = 500/873
w_obj_func_2b_C2 = svm.primal_stochastic_subgradient_descent_svm(S,T,gamma,C2)
print(w_obj_func_2b_C2[1])
w_2b_C2 = w_obj_func_2b_C2[0]
#TRAIN ERROR
error_train_2b_C2 = 0
i = 0
for train in S:
    prediction = svm.predict_svm(w_2b_C2,train[0])
    if prediction != train[1]:
        error_train_2b_C2 += 1/total_train_examples
    i += 1
print("Average train prediction error:")
print(error_train_2b_C2)
#TEST ERROR
error_test_2b_C2 = 0
i = 0
for test in S_test:
    prediction = svm.predict_svm(w_2b_C2,test)
    if prediction != y_test[i]:
        error_test_2b_C2 += 1/total_test_examples
    i += 1
print("Average test prediction error:")
print(error_test_2b_C2)
print("w found:")
print(w_2b_C2)

print("C = 700/873")
print("**************")
C3 = 700/873
w_obj_func_2b_C3 = svm.primal_stochastic_subgradient_descent_svm(S,T,gamma,C3)
print(w_obj_func_2b_C3[1])
w_2b_C3 = w_obj_func_2b_C3[0]
#TRAIN ERROR
error_train_2b_C3 = 0
i = 0
for train in S:
    prediction = svm.predict_svm(w_2b_C3,train[0])
    if prediction != train[1]:
        error_train_2b_C3 += 1/total_train_examples
    i += 1
print("Average train prediction error:")
print(error_train_2b_C3)
#TEST ERROR
error_test_2b_C3 = 0
i = 0
for test in S_test:
    prediction = svm.predict_svm(w_2b_C3,test)
    if prediction != y_test[i]:
        error_test_2b_C3 += 1/total_test_examples
    i += 1
print("Average test prediction error:")
print(error_test_2b_C3)
print("w found:")
print(w_2b_C3)



