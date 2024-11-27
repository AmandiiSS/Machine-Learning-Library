import scipy
import numpy

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


print("Exercise 3a:")
print("-------------")

C1 = 100/873
C2 = 500/873
C3 = 700/873

def dual_obj_func_ex_3a(alpha):
    sum_alpha_y_x = [0]*len(S[0][0])
    for i in range(len(S)):
        for k in range(len(S[i][0])):
            sum_alpha_y_x[k] += alpha[i]*S[i][1]*S[i][0][k]
    dot_product = sum([v * v for v in sum_alpha_y_x])
    return 0.5*dot_product - sum(alpha)

def constraint1(alpha):
    return sum(alpha[i] * S[i][1] for i in range(len(S)))

# def constraint2(alpha):
#    return -alpha

# def constraint3(alpha):
#      alpha_C = alpha
#      for i in range(len(alpha)):
#          alpha_C[i] = alpha[i] - C
#      return alpha_C


# Bounds: 0 <= alpha_i <= C
bounds1 = [(0, C1)] * len(S)
bounds2 = [(0, C2)] * len(S)
bounds3 = [(0, C3)] * len(S)

# Initial guess
alpha0 = [0] * len(S)

print("C = 100/873")
print("**************")

result = scipy.optimize.minimize(
    fun = dual_obj_func_ex_3a,
    x0 =alpha0,
    method="SLSQP",
    constraints=[{'type':'eq', 'fun':constraint1}], 
    bounds=bounds1
    )

print("Alpha opt:")
alpha_opt = result.x
#print(alpha_opt)

print("w opt:")
# Calculate w from alpha
w = [0] * len(S[0][0])
for i in range(len(S)):
    for k in range(len(S[i][0])):
        w[k] += alpha_opt[i] * S[i][1] * S[i][0][k]
print(w)



print("C = 500/873")
print("**************")

result = scipy.optimize.minimize(
    fun = dual_obj_func_ex_3a,
    x0 =alpha0,
    method="SLSQP",
    constraints=[{'type':'eq', 'fun':constraint1}], 
    bounds=bounds2
    )

print("Alpha opt:")
alpha_opt = result.x
#print(alpha_opt)

print("w opt:")
# Calculate w from alpha
w = [0] * len(S[0][0])
for i in range(len(S)):
    for k in range(len(S[i][0])):
        w[k] += alpha_opt[i] * S[i][1] * S[i][0][k]
print(w)



print("C = 700/873")
print("**************")

result = scipy.optimize.minimize(
    fun = dual_obj_func_ex_3a,
    x0 =alpha0,
    method="SLSQP",
    constraints=[{'type':'eq', 'fun':constraint1}], 
    bounds=bounds3
    )

print("Alpha opt:")
alpha_opt = result.x
#print(alpha_opt)

print("w opt:")
# Calculate w from alpha
w = [0] * len(S[0][0])
for i in range(len(S)):
    for k in range(len(S[i][0])):
        w[k] += alpha_opt[i] * S[i][1] * S[i][0][k]
print(w)






print("Exercise 3b:")
print("-------------")

gammas = [0.1,0.5,1,5,100]
Cs = [100/873, 500/873, 700/873]

for g in gammas:
    gamma = g
    print("Gamma = ", g)

    def gauss_kernel(x, y):
        x_numpy = numpy.array(x)
        y_numpy = numpy.array(y)
        return numpy.exp(-numpy.linalg.norm(x_numpy-y_numpy)**2/gamma)


    def dual_obj_func_ex_3b_gauss_kernel(alpha):
        total_sum = 0
        for i in range(len(S)):
            for j in range(len(S)):
                total_sum += alpha[i]*alpha[j]*S[i][1]*S[j][1]*gauss_kernel(S[i][0],S[j][0])
        return 0.5*total_sum - sum(alpha)
    
    def compute_bias(alpha_opt):
        bias_values = []
        # Loop through all support vectors (where alpha > 0)
        for i in range(len(S)):
            if alpha_opt[i] > 0:  # Only consider support vectors
                # Compute the kernel sum for the current support vector
                kernel_sum = 0
                for j in range(len(S)):
                    kernel_sum += alpha_opt[j] * S[j][1] * gauss_kernel(S[j][0], S[i][0])
                # Calculate the bias using the current support vector
                b = S[i][1] - kernel_sum
                bias_values.append(b)
        # Return the average bias value
        return numpy.mean(bias_values)
    
    def pred_funct(alpha_opt,x,b):
        kernel_sum = 0
        for i in range(len(S)):
            kernel_sum += alpha_opt[i] * S[i][1] * gauss_kernel(S[i][0], x)
        # Add the bias term
        decision_value = kernel_sum + b
        # Return the sign of the decision function (either -1 or +1)
        return 1 if decision_value > 0 else -1

    for c in Cs:
        print("C = ", c)
        print("----------------------------------")
        bounds_b = [(0, c)] * len(S)

        result = scipy.optimize.minimize(
            fun = dual_obj_func_ex_3b_gauss_kernel,
            x0 =alpha0,
            method="SLSQP",
            constraints=[{'type':'eq', 'fun':constraint1}], 
            bounds=bounds_b
            )
        
        print("Optim. done")
        
        alpha_opt = result.x
        b = compute_bias(alpha_opt)

        print("b calculated")
        
        total_test_examples = len(S_test)
        total_train_examples = len(S)

        #TRAIN ERROR
        error_train_3b = 0
        i = 0
        for train in S:
            prediction = pred_funct(alpha_opt,train[0],b)
            if prediction != train[1]:
                error_train_3b += 1/total_train_examples
            i += 1
        print("Average train prediction error:")
        print(error_train_3b)
        #TEST ERROR
        error_test_3b = 0
        i = 0
        for test in S_test:
            prediction = pred_funct(alpha_opt,test,b)
            if prediction != y_test[i]:
                error_test_3b += 1/total_test_examples
            i += 1
        print("Average test prediction error:")
        print(error_test_3b)
       



