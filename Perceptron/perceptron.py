import random

#Auxiliar functions
def dot(x, y):
    dot_result = 0
    for i in range(len(x)):
        dot_result += x[i]*y[i]
    return dot_result

def update_w(w,r,y,x):
    w_new = [0] * len(w)
    for i in range(len(w)):
        w_new[i] = w[i] + r*y*x[i]
    return w_new

#The examples of S consist in a list of [[x1 x2 ... xn 1], y], the last one is associated to the bias term
#Here y is either -1 or 1
#The weight vector is a list of [w1 w2 ... wn b]

def standard_perceptron(S,T,r):
    #len(S[0][0]) should be the lenght of [x1 x2 ... xn 1]
    w = [0]*len(S[0][0])
    for i in range(T):
        random.shuffle(S)
        for example in S:
            y = example[1]
            x = example[0]
            res = y*dot(w,x)
            if res <= 0:
                #update w
                w = update_w(w,r,y,x)
    return w

def voted_perceptron(S,T,r):
    voted_w = []
    #len(S[0][0]) should be the lenght of [x1 x2 ... xn 1]
    w = [0]*len(S[0][0])
    m = 0
    Cm = 1
    for i in range(T):
        random.shuffle(S)
        for example in S:
            y = example[1]
            x = example[0]
            res = y*dot(w,x)
            if res <= 0:
                voted_w.append([w,Cm])
                #update w
                w = update_w(w,r,y,x)
                m += 1
                Cm = 1
            else: 
                Cm += 1
    return voted_w

def averaged_perceptron(S,T,r):
    #len(S[0][0]) should be the lenght of [x1 x2 ... xn 1]
    w = [0]*len(S[0][0])
    a = [0]*len(S[0][0])
    for i in range(T):
        random.shuffle(S)
        for example in S:
            y = example[1]
            x = example[0]
            res = y*dot(w,x)
            if res <= 0:
                #update w
                w = update_w(w,r,y,x)
            for j in range(len(a)):
                a[j] += w[j]
    return a

def predict_perceptron(w,x):
    if dot(w,x) <= 0:
        return -1
    else:
        return 1
    
def predict_voted_perceptron(voted_w,x):
    pred = 0
    for wc in voted_w:
        if dot(wc[0],x) <= 0:
            pred += -wc[1]
        else:
            pred += wc[1]
    if pred <= 0:
        return -1
    else:
        return 1
    
def predict_averaged_perceptron(a,x):
    if dot(a,x) <= 0:
        return -1
    else:
        return 1