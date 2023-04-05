'''
# Step53 모델 저장 및 읽어오기
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.dataloaders import DataLoader
from dezero.models import MLP

if __name__ == '__main__':
    # 53.1 넘파이의 save 함수와 load 함수
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    data = {'x1': x1, 'x2': x2}
    np.savez('test.npz', **data)

    arrays = np.load('test.npz')
    x1 = arrays['x1']
    x2 = arrays['x2']
    print(x1)
    print(x2)

    # 53.2 Layer 클래스의 매개변수를 평평하게
    layer = dezero.Layer()

    l1 = dezero.Layer()
    l1.p1 = dezero.Parameter(np.array(1))

    layer.l1 = l1
    layer.p2 = dezero.Parameter(np.array(2))
    layer.p3 = dezero.Parameter(np.array(3))

    params_dict = {}
    layer._flatten_params(params_dict)
    print(params_dict)

    # 53.3 Layer 클래스의 save 함수와 load 함수
    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)

    model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
    optimizer = optimizers.SGD().setup(model)

    if os.path.exists('my_mlp.npz'):
        model.load_weights('my_mlp.npz')

    for epoch in range(max_epoch):
        sum_loss= 0
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)

        print(f"Epoch: {epoch + 1}, Loss: {sum_loss / len(train_set):.4f}")

    model.save_weights('my_mlp.npz')