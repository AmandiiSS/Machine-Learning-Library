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

print("Y:")
print(y)
print("X:")
print(X)
print("The matrix and vector calculations of the optimal have been calculated with MATLAB.")