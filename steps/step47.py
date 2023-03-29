'''
# Step47 소프트맥스 함수와 교차 엔트로피 오차
- 
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

import dezero.functions as F
from dezero import Variable, as_variable
from dezero.models import MLP


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

if __name__ == '__main__':
    # 47.1 슬라이스 조작 함수
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.get_item(x, 1)
    print(y)

    y.backward()
    print(x.grad)

    y = x[1]
    print(y)

    y = x[:,2]
    print(y)

    # 47.2 소프트맥스 함수
    model = MLP((10, 3))
    x = np.array([[0.2, -0.4]])
    y = model(x)
    p = softmax1d(y)
    print(y)
    print(p)

    # 47.3 교차 엔트로피 오차
    x = Variable(np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
    t = Variable(np.array([2, 0, 1, 0]))
    y = model(x)
    loss = F.softmax_cross_entropy(y, t)
    print(loss)
    