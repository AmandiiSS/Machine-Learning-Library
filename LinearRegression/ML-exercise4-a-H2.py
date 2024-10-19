import gradientdescent

w0 = [0]*8
r = 0.01498
#r = 0.5
X = []
y = []


CSVfile = 'concrete/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_train = line.strip().split(',')
        xi = [1]
        for i in range(len(example_train)-1):
            xi.append(float(example_train[i]))
        y.append(float(example_train[len(example_train)-1]))
        X.append(xi)

max_it = 10000
result = gradientdescent.GradientDescent(w0,X,y,r,max_it)
print("r = ", r)
print("Convergence: ", result[1])
print("Num of iterations: ", result[2])
print("w = ", result[0])
print("Function values:")
print(result[3])

X = []
y = []

CSVfile = 'concrete/test.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_test = line.strip().split(',')
        xi = [1]
        for i in range(len(example_test)-1):
            xi.append(float(example_test[i]))
        y.append(float(example_test[len(example_test)-1]))
        X.append(xi)

function_values_test = 0
for i in range(len(y)):
    function_values_test += 0.5*pow((y[i] - gradientdescent.dot(result[0], X[i])),2)
print("Cost function value of the test data:")
print(function_values_test)
    