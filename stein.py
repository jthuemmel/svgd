import numpy as np
import scipy.spatial.distance as distance

def svgd(x0, dlogp, step_size = 1e-3, iterations = 1000, alpha = 0.9, h = -1):
    #x0 = initial particles (q distributed)
    #dlogp = first derivative of log - probability (function!)
    #step_size = step of gradient descent
    #iterations = number of grad descent iterations
    #alpha = weight of history vs current gradient
    # h = bandwidth of kernel, if < 0 use median trick from paper
    
    x = x0.copy()
    
    epsilon = 1e-6
    hist_grad = 0
        
    for l in range(iterations):
        
        if x0.shape[0] != 1:
            dist_mat = distance.squareform(distance.pdist(x)) **2 #use scipy package to calculate pairwise euclidean distance mat

            if h < 0 : #as suggested in the paper, calculate bandwith like so
                h = np.sqrt(0.5*np.median(dist_mat) / np.log(x.shape[0]+1))

            kxy = np.exp(-dist_mat / h**2 /2) #rbf kernel formula

            dkxy = -np.matmul(kxy , x) #first part of derivative of kxy
            for i in range(x.shape[1]): #second part of derivative of kxy
                dkxy[:,i] += x[:,i] * np.sum(kxy,axis=1)
            dkxy /= h**2
            
            x_grad = (1/x0.shape[0])*(np.matmul(kxy,dlogp(x)) + dkxy) #formula from Algorithm 1
            
        else:
            x_grad = dlogp(x) #formula from Algorithm 1 without the kernel term 
         
        
        
        if l == 0:
            hist_grad = x_grad**2 #here we have no history yet
        else:
            hist_grad = alpha * hist_grad + (1 - alpha) * (x_grad ** 2) #weighted sum of historical and current gradient
            
        adj_grad = x_grad / (np.sqrt(hist_grad) + epsilon) #adagrad formula
        
        x = x + step_size * adj_grad #gradient descent step
    
    return x