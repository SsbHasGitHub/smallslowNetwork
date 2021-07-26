import numpy as np
import Layer
import Network
import Loss
import matplotlib.pyplot as plt
import tool
#from sklearn.datasets import  fetch_openml
#mnist = fetch_openml('mnist_784',version=1)
#print(mnist.keys())
inputs=np.linspace(-3,3,40).reshape(-1,1)
outputs=np.power(inputs,3)

layer1=Layer.linear(np.random.randn(1,4),np.zeros((1,4)))
layer2=Layer.linear(np.random.randn(4,4),np.zeros((1,4)))
layer3=Layer.linear(np.random.randn(4,4),np.zeros((1,4)))
layer4=Layer.linear(np.random.randn(4,3),np.zeros((1,1)))
layer5=Layer.linear(np.random.randn(4,1),np.zeros((1,1)))

net=Network.net()
net.add([layer1,Layer.relu(),layer2,Layer.relu(),layer3,Layer.relu(),layer4,Layer.softmax()])


for i in range(800):
    out=net.forward(inputs)
    net.backward(Loss.crossEntropy(out,outputs))
    net.update(0.000001)
    #print(net.backward(Loss.crossEntropy(pre=out,targets=3)  ))
    #print(tool.score(outputs,out))

    print("epoch=",i)
out=net.forward((inputs))


plt.plot(inputs,out,'b')
plt.plot(inputs,outputs,'r')
plt.show()






