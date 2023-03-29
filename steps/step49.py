'''
# Step49 Dataset 클래스와 전처리
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
    # 49.1 Dataset 클래스 구현
    train_set = dezero.datasets.Spiral(train=True)
    print(train_set[0])
    print(len(train_set))

    # 49.3 데이터 이어 붙이기
    batch_index = [0, 1, 2]
    batch = [train_set[i] for i in batch_index]
    x = np.array([example[0] for example in batch])
    t = np.array([example[1] for example in batch])

    print(x.shape)
    print(t.shape)

    # 학습 코드
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0
    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)

    train_set = dezero.datasets.Spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    losses = []
    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            batch_index = index[i * batch_size: (i + 1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print(f"epoch {epoch + 1}, loss {avg_loss:.2f}")
        losses.append(avg_loss)

    # 49.5 데이터셋 전처리
    def f(x):
        y = x / 2.0
        return y
    train_set = dezero.datasets.Spiral(transform=f)
    x = np.array([example[0] for example in train_set])
    t = np.array([example[1] for example in train_set])
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x[np.where(t==0)[0], 0], x[np.where(t==0)[0], 1], 'ro')
    plt.plot(x[np.where(t==1)[0], 0], x[np.where(t==1)[0], 1], 'bx')
    plt.plot(x[np.where(t==2)[0], 0], x[np.where(t==2)[0], 1], 'g^')

    plt.subplot(122)
    plt.plot(np.arange(max_epoch) + 1, losses)
    plt.show()