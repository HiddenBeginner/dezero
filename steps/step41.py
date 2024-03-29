'''
# Step41 행렬의 곱
- 
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':
    # 41.3 행렬 곱의 역전파
    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W)
    y.backward()

    print(x.grad.shape)
    print(W.grad.shape)