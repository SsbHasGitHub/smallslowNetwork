import numpy as np
class linear(object):
    name='linear'

    def __init__(self,initial_W, initial_b):
        self.W=initial_W
        self.b=initial_b

    def forward(self,inputs):
        self.inputs=inputs
        return  inputs @ self.W +self.b

    def backward(self,grad):
        self.grad=grad
        self.grads_change=(grad @ self.inputs ).T
        self.b_change=np.sum(grad,axis=1).reshape(1,-1)
        return self.W @ grad

class relu(object):
    name='relu'
    def __init__(self):
        pass

    def forward(self,inputs):
        self.derivative=np.where(inputs>0,1,0)
        return np.where(inputs>0,inputs,0)

    def backward(self,grad):

        return self.derivative.T * grad

class softmax(object):
    name='softmax'
    def __init__(self):
        pass

    def forward(self,inputs):
        sum=np.sum(np.exp(inputs),axis=1).reshape(-1,1)
        return np.exp(inputs)/sum

    def backward(self,grad):
        return grad
