'''
# Step23 패키지로 정리

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
    # 23.2 코어 클래스로 옮기기
    x = Variable(np.array(1.0))
    print(x)

    # 23.5 dezero 임포트하기
    x = Variable(np.array(1.0))
    y = (x + 3) ** 2
    y.backward()

    print(y)
    print(x.grad)
