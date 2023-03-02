'''
# Step34 Sin 함수 고차 미분
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
    # 34.3 sin 함수 고차 미분
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)
    
    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad)
        
    x = Variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data]

    for i in range(3):
        logs.append(x.grad.data)
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
    
    labels = ['y=sin(x)', "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc='lower right')
    plt.show()
