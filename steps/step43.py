'''
# Step43 신경망
- 
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':
    # 43.2 비선형 데이터셋
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # 43.4 신경망 구현
    x, y = Variable(x), Variable(y)
    I, H, O = 1, 10, 1
    W1 = Variable(0.01 * np.random.randn(I, H))
    b1 = Variable(np.zeros(H))
    W2 = Variable(0.01 * np.random.randn(H, O))
    b2 = Variable(np.zeros(O))

    def predict(x):
        y = F.linear(x, W1, b1)
        y = F.sigmoid(y)
        y = F.linear(y, W2, b2)
        return y

    lr = 0.2
    iters = 10000
    
    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        loss.backward()
        for param in [W1, b1, W2, b2]:
            param.data -= lr * param.grad.data
            param.cleargrad()

        if i % 1000 == 0:
            print(loss)
        
    plt.figure()
    x_test = Variable(np.linspace(0, 1, 25, dtype=np.float64).reshape(-1, 1))
    y_pred = predict(x_test)
    plt.plot(x.data, y.data, 'o')
    plt.plot(x_test.data, y_pred.data)
    plt.show()