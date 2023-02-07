'''
# Step2 변수를 낳는 함수

- 함수: 어떤 변수로부터 다른 변수로의 대응 관계를 정한 것
- 계산 그래프: 변수와 함수를 노드로 사용하여 계산 과정을 표현한 그림

**Function** 클래스
- Base 클래스로서 모든 함수에 공통되는 기능을 구현한다. 
- 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현
- Variable 인스턴스를 입력받아 Variable 인스턴스를 출력
- Variable 인스턴스의 실제 데이터는 인스턴스 속성 data에 저장되어 있다.
- forward()의 raise NotImplementedError는 사용자에게 "이 메서드는 상속하여 구현해야 한다"는 것을 알려준다

'''
class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2


if __name__ == '__main__':
    import numpy as np

    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)
