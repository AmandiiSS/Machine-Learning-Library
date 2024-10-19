
def dot(x, y):
    dot_result = 0
    for i in range(len(x)):
        dot_result += x[i]*y[i]
    return dot_result

def norm2 (x):
    norm_squared = 0
    for element in x:
        norm_squared += pow(element,2)
    return pow(norm_squared,0.5)

def subtract(x, y):
    a = [0]*len(x)
    for i in range(len(x)):
        a[i] = x[i] - y[i]
    return a


def GradientDescent (w0,X,y,r,max_it):
    converge = False
    it = 0
    function_values = []

    while (not converge) and (it<max_it):
        it += 1
        #Compute the gradient
        dJ = []
        for j in range(len(w0)):
            dJj = 0
            for i in range(len(y)):
                xi = X[i]
                dJj -= (y[i] - dot(w0, xi))*X[i][j]
            dJ.append(dJj)
        #Update w
        w = []
        for i in range(len(w0)):
            w.append(w0[i] -r*dJ[i])
        #Calculate LMS
        func_val_i = 0
        for i in range(len(y)):
            func_val_i += 0.5*pow((y[i] - dot(w, X[i])),2)
        function_values.append(func_val_i)
        #See if we want to stop or not
        normww0 = norm2(subtract(w,w0))
        if normww0 <1e-6:
            converge = True
        else:
            w0 = w

    return [w0,converge, it, function_values]

#REVISAR
def StochGradientDescent (w0,X,y,r):
    stop = False
    while(not stop):
        for i in len(y):
            w = []
            for j in range(len(w0)):
                xi = np.array(X[i])
                w.append(w0[j] +r*(y[i] - np.dot(w0, xi))*X[i][j])
        #See if we want to stop or not
        normww0 = np.linalg.norm(np.array(w)-np.array(w0))
        if normww0 <10e-6:
            stop = True
        else:
            w0 = w
    return w0