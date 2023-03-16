'''
# Step39 합계 함수

'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':
    # 39.2 sum 함수 구현
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)
    
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    # 39.3 axis와 keepdims
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)