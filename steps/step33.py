'''
# Step33 뉴턴 방법으로 푸는 최적화(자동 계산)
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


if __name__ == '__main__':
    # 33.1 2차 미분 계산하기
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad)
    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad)

    # 33.2 뉴턴 방법을 활용한 최적화
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx = x.grad
        x.cleargrad
        gx.backward()
        gx2 = x.grad

        x.data -= gx.data / gx2.data
