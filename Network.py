#所有数据都要用float，都要是二维数组

import numpy as np
class net(object):
    Layers=[]
    def __init__(self,name=None):
        self.name=name

    def add(self,layer):
        self.Layers.extend(layer)
        return

    def drop(self):
        self.Layers=self.Layers[0:-1]

    def forward(self,inputs):
        for i in self.Layers:
            inputs=i.forward(inputs)
        return inputs

    def backward(self,grad):
        for i in reversed(self.Layers):
            grad=i.backward(grad)
        return grad

    def update(self,learning_rate,optimizer=None):
        for i in self.Layers:
            if i.name=='linear':
               i.W-=i.grads_change*learning_rate
               i.b-=i.b_change*learning_rate


