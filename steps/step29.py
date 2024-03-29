'''
# Step29 뉴턴 방법으로 푸는 최적화 (수동 계산)
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


if __name__ == '__main__':
    # 29.2 뉴턴 방법을 활용한 최적화 구현
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)
        
        y = f(x)
        x.cleargrad()
        y.backward()

        x.data -= x.grad / gx2(x.data)

