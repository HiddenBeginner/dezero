'''
# Step12 가변 길이 인수(개선편)
'''
import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"<{type(data)}>는 지원하지 않습니다.")

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
            
            if input.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
        
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()


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
    
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)


if __name__ == '__main__':
    # 12.1 첫 번째 개선: 함수를 사용하기 쉽게
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    f = Add()
    y = f(x0, x1)
    print(y.data)

    # 12.2 두 번쨰 개선: 함수를 구현하기 쉽도록
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    f = Add()
    y = f(x0, x1)
    print(y.data)

    # 12.3 add 함수 구현
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)
    