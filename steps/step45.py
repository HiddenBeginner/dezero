'''
# Step44 계층을 모아놓는 계층
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
from dezero import Model, Variable
from dezero.layers import Layer


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


if __name__ == '__main__':
    # 45.1 Layer 클래스 확장
    model = Layer()
    model.l1 = L.Linear(5)
    model.l2 = L.Linear(3)
    
    def predict(x):
        y = model.l1(x)
        y = F.sigmoid(y)
        y = model.l2(y)
        return y

    # 모든 매개변수에 접근
    for p in model.params():
        print(p)

    # 모든 매개변수의 기울기 재설정
    model.cleargrads()
    
    # 45.3 Model을 사용한 문제 해결
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)
    
    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = TwoLayerNet(hidden_size, 1)
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()
        
        for p in model.params():
            p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)

    plt.figure()
    x_test = Variable(np.linspace(0, 1, 25, dtype=np.float64).reshape(-1, 1))
    y_pred = model(x_test)
    plt.plot(x.data, y.data, 'o')
    plt.plot(x_test.data, y_pred.data)
    plt.show()
