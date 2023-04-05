'''
제 5고지 DeZero의 도전
# Step52 GPU 지원
- GPU <-> CPU 전송 과정이 병목으로 작용 => 데이터 전송 횟수를 최소로
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import time

import cupy as cp
import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.dataloaders import DataLoader
from dezero.models import MLP

if __name__ == '__main__':
    # 52.1 쿠파이 설치 및 사용 방법
    x = cp.arange(6).reshape(2, 3)
    print(x)
    print("Device: ", x.device)

    y = x.sum(axis=1)
    print(y)

    # 넘파이 배열 -> 쿠파이 배열
    n = np.array([1, 2, 3])
    c = cp.asarray(n)
    print(type(c) == cp.ndarray)

    # 쿠파이 배열 -> 넘파이 배열
    c = cp.array([1, 2, 3])
    n = cp.asnumpy(c)
    print(type(n) == np.ndarray)

    # cp.get_array_module: 주어진 데이터에 적합한 모듈 반환. 
    # 넘파이면 넘파이 모듈, 쿠파이면 쿠파이 모듈
    # 넘파이/쿠파이에 모두 대응하는 코드를 작성할 수 있음
    x = np.array([1, 2, 3])
    xp = cp.get_array_module(x)
    print(xp == np)

    x = cp.array([1, 2, 3])
    xp = cp.get_array_module(x)
    print(xp == cp)
    
    # 52.5 GPU로 MNIST 학습하기
    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)

    model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
    optimizer = optimizers.SGD().setup(model)

    if dezero.cuda.gpu_enable:
        print(1)
        train_loader.to_gpu()
        model.to_gpu()

    for epoch in range(max_epoch):
        start = time.time()
        sum_loss= 0
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)

        elapsed_time = time.time() - start
        print(f"Epoch: {epoch + 1}, Loss: {sum_loss / len(train_set):.4f}, Time: {elapsed_time:.4f}[sec]")