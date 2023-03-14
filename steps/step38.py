'''
# Step38 형상 변환 함수

## 38.1
- reshape 함수의 역전파는 gy를 reshape 역연산을 해서 보내주면 된다.
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':
    # 38.1 reshape 함수 구현
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.reshape(x, (6,))
    print(y)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6, ))
    y.backward(retain_grad=True)
    print(y.grad)
    print(x.grad)

    # 38.2 Variable에서 reshape 사용하기
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.reshape((2, 3))
    print(y)
    y = x.reshape(2, 3)
    print(y)

    # 38.3 행렬의 전치
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward(retain_grad=True)
    print(y.grad)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.transpose()
    print(y)
    y = x.T
    print(y)
