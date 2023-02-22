'''
# Step24 복잡한 함수의 미분

- 모듈: 파이썬 파일. 특히 다른 파이썬 프로그램에서 임포트하여 사용하는 것을 가정하고 만들어진 파이썬 파일
- 패키지: 여러 모듈을 하나의 디렉토리에 묶은 것
- 라이브러리: 여러 패키지를 묶은 것. 하나 이상의 디렉토리로 구성됌

'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable

if __name__ == '__main__':
    # 24.1 Sphere 함수
    def sphere(x, y):
        z = x ** 2 + y ** 2
        return z

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = sphere(x, y)
    z.backward()
    print(x.grad, y.grad)

    # 24.2 matyas 함수
    def matyas(x, y):
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return z

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = matyas(x, y)
    z.backward()
    print(x.grad, y.grad)

    # 24.3 Goldstein-Price 함수
    def goldstein(x, y):
        z = ((
            1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
        ) * (
           30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
        ))
        return z

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    z.backward()
    print(x.grad, y.grad)
