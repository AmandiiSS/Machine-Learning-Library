import perceptron

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
print("-------------")
w_2a = perceptron.standard_perceptron(S,10,1)
print("learned w:")
print(w_2a)

total_examples = len(S)
error_test_2a = 0
i = 0
for test in S_test:
    prediction = perceptron.predict_perceptron(w_2a,test)
    if prediction != y_test[i]:
        error_test_2a += 1/total_examples
    i += 1
print("Average prediction error:")
print(error_test_2a)


print("Exercise 2b:")
print("-------------")
voted_w_2b = perceptron.voted_perceptron(S,10,1)
print("learned w and counts:")
for w_cm in voted_w_2b:
    print(w_cm[0])
    print(w_cm[1])

error_test_2b = 0
i = 0
for test in S_test:
    prediction = perceptron.predict_voted_perceptron(voted_w_2b,test)
    if prediction != y_test[i]:
        error_test_2b += 1/total_examples
    i += 1
print("Average prediction error:")
print(error_test_2b)


print("Exercise 2c:")
print("-------------")
a_2c = perceptron.averaged_perceptron(S,10,1)
print("learned weight vector a:")
print(a_2c)

error_test_2c = 0
i = 0
for test in S_test:
    prediction = perceptron.predict_averaged_perceptron(a_2c,test)
    if prediction != y_test[i]:
        error_test_2c += 1/total_examples
    i += 1
print("Average prediction error:")
print(error_test_2c)
