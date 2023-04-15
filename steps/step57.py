'''
# Step57 conv2d 함수와 pooling 함수
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable
from dezero.utils import pair

if __name__ == '__main__':
    # 57.2 conv2d 함수 구현
    ## im2col 함수
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
    print(col1.shape)

    x2 = np.random.rand(10, 3, 7, 7)
    kernel_size = (5, 5)
    stride = (1, 1)
    pad = (0, 0)
    col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
    print(col2.shape)

    ## pair 함수
    print(pair(1))
    print(pair((1, 2)))
    
    ## conv2d_simple 함수
    N, C, H, W = 1, 5, 15, 15
    OC, KH, KW = 8, 3, 3
    x = Variable(np.random.randn(N, C, H, W))
    W = np.random.randn(OC, C, KH, KW)
    y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
    y.backward()
    print(y.shape)
    print(x.grad.shape)