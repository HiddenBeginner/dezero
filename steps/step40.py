'''
# Step40 브로드캐스트 함수
- 원소 복사가 일어나면 기울기를 합치게 된다.
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

from dezero import Variable

if __name__ == '__main__':
    # 40.1 NumPydml broadcast_to 함수
    x = np.array([1, 2, 3])
    y = np.broadcast_to(x, (2, 3))
    print(y)
    
    # 40.3 브로드캐스트 대응
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    print(y)

    y.backward()
    print(x0.grad)
    print(x1.grad)