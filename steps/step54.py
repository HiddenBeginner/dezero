'''
# Step54 드롭아웃과 테스트 모드
- 역드롭아웃은 테스트시 아무것도 하지 않기 때문에 학습 중 드랍아웃 비율 동적으로 변경할 수 있다.
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np

import dezero.functions as F
from dezero import test_mode

if __name__ == '__main__':
    # 54.1 드롭아웃이란
    dropout_ratio = 0.6
    x = np.ones(10)
    # 학습 시
    mask = np.random.rand(10) > dropout_ratio
    y = x * mask
    print(y)
    # 테스트 시
    scale = 1 - dropout_ratio
    y = x * scale
    print(y)

    # 54.2 역 드롭아웃
    # 학습 시
    scale = 1 - dropout_ratio
    y = x * mask / scale
    print(y)
    # 테스트 시
    y = x
    print(y)

    # 54.4 드롭아웃 구현
    x = np.ones(5)
    print(x)
    # 학습시
    y = F.dropout(x)
    print(y)
    # 테스트 시
    with test_mode():
        y = F.dropout(x)
        print(y)