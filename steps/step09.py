'''
# Step9 함수를 더 편리하게
'''
import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


if __name__ == '__main__':
    # 9.1 파이썬 함수로 이용하기
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    # 9.2 backward 메서드 간소화
    x =  Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)

    # 9.3 ndarray만 취급하기
    x = Variable(np.array(1.0))
    x = Variable(None)
    x = Variable(1.0)