'''
# Step36 고차 미분 이외의 용도
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

from dezero import Variable

if __name__ == '__main__':
    # 36.1 double backprop 용도
    x = Variable(np.array(2.0))
    y = x ** 2
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()

    z = gx ** 3 + y
    z.backward()
    print(x.grad)