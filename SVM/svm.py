import random

#Auxiliar functions
def dot(x, y):
    dot_result = 0
    for i in range(len(x)):
        dot_result += x[i]*y[i]
    return dot_result

def update_w(w,gamma_t,C,N,y,x):
    w_new = [0] * len(w)
    for i in range(len(w)-1):
        w_new[i] = w[i] - gamma_t*w[i] + gamma_t*C*N*y*x[i]
    w_new[len(w)-1] = w[len(w)-1] + gamma_t*C*N*y*x[len(w)-1]
    return w_new

def update_w_else(w,gamma_t):
    w_new = [0] * len(w)
    for i in range(len(w)-1):
        w_new[i] = (1-gamma_t)*w[i]
    return w_new


#The examples of S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term
#Here y is either -1 or 1
#The weight vector is a list of [w1 w2 ... wn b]

def primal_stochastic_subgradient_descent_svm(S,T,gamma,C):
    #len(S[0][0]) should be the lenght of [x1 x2 ... xn 1]
    #gamma is a vector with all the learning rates gamma_t
    obj_func_all_epoch = []
    N = len(S)
    w = [0]*len(S[0][0])
    for i in range(T):
        random.shuffle(S)
        for example in S:
            y = example[1]
            x = example[0]
            res = y*dot(w,x)
            if res <= 1:
                #update w
                w = update_w(w,gamma[i],C,N,y,x)
            else:
                w = update_w_else(w,gamma[i])
        penalization_term = 0
        for example in S:
            penalization_term += max(0,1-example[1]*dot(w,example[0]))
        obj_funct_T = 0.5*dot(w,w)+C*penalization_term
        obj_func_all_epoch.append(obj_funct_T)
    return [w,obj_func_all_epoch]


def predict_svm(w,x):
    if dot(w,x) <= 0:
        return -1
    else:
        return 1
    
    
