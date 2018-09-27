import numpy as np
import time
np.random.seed(1)

class neural_network:
    def __init__(self,layers_dims):
       self.layers_dims=layers_dims        
    def initialize_parameters(self):
       np.random.seed(1)
       parameters={}
       L = len(self.layers_dims)
       for i in range(1,L):
           parameters["W" + str(i)]=np.random.randn(self.layers_dims[i],self.layers_dims[i-1])*0.01
           parameters["b" + str(i)]=np.zeros((self.layers_dims[i],1))
       return parameters    

    def linear_forward(self,A,W,b):
        Z=np.dot(W,A)+b
        cache = (A,W,b)
        return Z,cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = self.relu(Z)
        caches = (cache, activation_cache)
        return A, caches

    def sigmoid(self,Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self,Z):
        A = np.maximum(0,Z)    
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache


    def L_model_forward(self,X,parameters): # X is input size i.e.,no of features and no of training examples(m)
        caches = []
        A = X
        L = len(parameters)//2

        for l in range(1,L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation="relu")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], activation="sigmoid")
        caches.append(cache)
        return AL,caches   
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)     
        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
       # print(m)
        dW = 1./m*np.dot(dZ, A_prev.T)
       # print(str(dW.shape) + " " + str(W.shape) )
        assert (dW.shape == W.shape)
        
        db = 1./m*np.sum(dZ, axis=1, keepdims=True)
        assert (db.shape == b.shape)
    
        dA_prev = np.dot(W.T, dZ)
        assert (dA_prev.shape == A_prev.shape)
        
        
        return dA_prev, dW, db
    
    def linear_activation_backward(self,dA,cache,activation):

        linear_cache,activation_cache=cache

        if activation=="sigmoid":
            dZ= self.sigmoid_backward(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dZ,linear_cache)
        
        if activation=="relu":
            dZ=self.relu_backward(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dZ,linear_cache)
        
        return dA_prev,dW,db
    
    def L_model_backward(self,AL,Y,caches):

        grads={}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
        #tic = time.time()
        current_cache = caches[L-1]
        grads["dA" + str(L-1)],grads["dW" + str(L)],grads["db" + str(L)] = self.linear_activation_backward(dAL,current_cache,activation="sigmoid")
        #toc = time.time()
        #print(str(1000*(toc-tic)))
        
        for l in reversed(range(L-1)):
            tic = time.time()
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            toc=time.time()
            #print(str(l) + " " + str(1000*(toc-tic)),end='\n')
        return grads
    
    def update_parameters(self,parameters,grads,learning_rate):

        L = len(parameters) // 2

        for l in range(L):
           # print(grads["dW" + str(l+1)])
            parameters["W" + str(l+1)] -=learning_rate*grads["dW" + str(l+1)]
            #print(grads["db" + str(l+1)])
            parameters["b" + str(l+1)] -= learning_rate*grads["db" + str(l+1)]
        return parameters
    

    def relu_backward(self,dA,cache):
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ

    
    def sigmoid_backward(self,dA,cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s) 
        #print(str(Z.shape) + " " + str(dZ.shape))
        assert (dZ.shape == Z.shape)
        return dZ