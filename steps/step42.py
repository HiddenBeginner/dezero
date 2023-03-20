'''
# Step42 선형회귀
- 
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':
    # 42.1 토이 데이터셋
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)

    # 42.3 선형 회귀 구현
    x, y = Variable(x), Variable(y)
    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        y = F.matmul(x, W) + b
        return y

    def mean_squared_error(x0, x1):
        diff = x0 - x1
        return F.sum(diff ** 2) / len(diff)

    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()
        
        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        print(W, b, loss)
