'''
# Step48 다중 클래스 분류
- 
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

if __name__ == '__main__':
    # 48.1 스파이럴 데이터셋
    x, t = dezero.datasets.get_spiral(train=True)
    print(x.shape)
    print(t.shape)

    # 48.2 학습 코드
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0
    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)

    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    losses = []
    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            batch_index = index[i * batch_size: (i + 1) * batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print(f"epoch {epoch + 1}, loss {avg_loss:.2f}")
        losses.append(avg_loss)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x[np.where(t==0)[0], 0], x[np.where(t==0)[0], 1], 'ro')
    plt.plot(x[np.where(t==1)[0], 0], x[np.where(t==1)[0], 1], 'bx')
    plt.plot(x[np.where(t==2)[0], 0], x[np.where(t==2)[0], 1], 'g^')

    plt.subplot(122)
    plt.plot(np.arange(max_epoch) + 1, losses)
    plt.show()