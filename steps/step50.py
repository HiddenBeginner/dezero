'''
# Step50 미니배치를 뽑아주는 DataLoader
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
from dezero.dataloaders import DataLoader
from dezero.datasets import Spiral
from dezero.models import MLP


class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()
        self.cnt += 1
        return self.cnt

if __name__ == '__main__':
    # 50.1 반복자란?
    t = [1, 2, 3]
    x = iter(t)
    print(type(x))  # <class 'list_iterator'>
    print(next(x))
    print(next(x))
    print(next(x))

    obj = MyIterator(5)
    for x in obj:
        print(x)

    # 50.2 DataLoader 사용하기
    batch_size = 10
    max_epoch = 1

    train_set = Spiral(train=True)
    test_set = Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    for epoch in range(max_epoch):
        for x, t in train_loader:
            print(x.shape, t.shape)
            break

        for x, t in test_loader:
            print(x.shape, t.shape)
            break

    # 50.3 accuracy 함수 구현하기
    y = np.array([[0.2, 0.8, 0.0], [0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    t = np.array([1, 2, 0])
    acc = F.accuracy(y, t)
    print(acc)

    # 50.4 스파이럴 데이터셋 학습 코드
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    losses = []
    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print(f"Epoch {epoch+1}")
        print(f"train loss {sum_loss / len(train_set):.2f}, acc: {sum_acc / len(train_set):.4f}")

        sum_loss, sum_acc = 0, 0
        with dezero.no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

        print(f"train loss {sum_loss / len(test_set):.2f}, acc: {sum_acc / len(test_set):.4f}")