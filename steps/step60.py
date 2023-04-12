'''
# Step60 LSTM과 데이터 로더
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
import dezero.layers as L
from dezero import Model


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


if __name__ == '__main__':
    # 60.1 시계열 데이터용 데이터 로더
    train_set = dezero.datasets.SineCurve(train=True)
    dataloader = dezero.dataloaders.SeqDataLoader(train_set, batch_size=3)
    x, t = next(dataloader)
    print(x)
    print('===========')
    print(t)

    # 60.2 LSTM 계층 구현
    max_epoch = 100
    batch_size = 30
    hidden_size = 100
    bptt_length = 30
    train_set = dezero.datasets.SineCurve(train=True)
    dataloader = dezero.dataloaders.SeqDataLoader(train_set, batch_size=batch_size)
    seqlen = len(train_set)

    model = BetterRNN(hidden_size, 1)
    optimizer = dezero.optimizers.SGD(lr=0.0001).setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0
        
        for x, t in train_set:
            x = x.reshape(1, 1)
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1

            if count % bptt_length == 0 or count == seqlen:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()

        avg_loss = float(loss.data) / count
        print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()
    pred_list = []

    with dezero.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data))
    
    plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
    plt.plot(np.arange(len(xs)), pred_list, label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
