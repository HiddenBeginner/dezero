'''
# Step44 매개변수를 모아두는 계층
- 
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Parameter, Variable
from dezero.layers import Layer

if __name__ == '__main__':
    # 44.2 Layer 클래스 구현
    layer = Layer()
    layer.p1 = Parameter(np.array(1))
    layer.p2 = Parameter(np.array(2))
    layer.p3 = Variable(np.array(3))
    layer.p4 = 'test'

    print(layer._params)
    print('---------------')
    for name in layer._params:
        print(name, layer.__dict__[name])
    
    # 44.4 Layer를 이용한 신경망 구현
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)
    
    l1 = L.Linear(10)
    l2 = L.Linear(1)

    def predict(x):
        y = l1(x)
        y = F.sigmoid(y)
        y = l2(y)
        return y

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        l1.cleargrads()
        l2.cleargrads()
        loss.backward()
        
        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(loss)

    plt.figure()
    x_test = Variable(np.linspace(0, 1, 25, dtype=np.float64).reshape(-1, 1))
    y_pred = predict(x_test)
    plt.plot(x.data, y.data, 'o')
    plt.plot(x_test.data, y_pred.data)
    plt.show()
