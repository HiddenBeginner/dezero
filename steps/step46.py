'''
# Step46 Optimizer로 수행하는 매개변수 갱신
- 
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable, optimizers
from dezero.models import MLP

if __name__ == '__main__':
    # 46.3 SGD 클래스을 사용한 문제 해결
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)
    
    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = MLP((hidden_size, 1))
    optimizer = optimizers.MomentumSGD(lr)
    optimizer.setup(model)
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()
        optimizer.update()
        
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
