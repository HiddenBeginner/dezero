'''
# Step17 메모리 관리와 순환 참조

## 17.1 메모리 관리

파이썬은 크게 두 가지 방식으로 메모리 관리를 한다.
- 참조 카운트: 참조 수 세는 방식
- 가비지 컬렉션: 세대를 기준으로 쓸모 없어진 객체를 회수

## 17.2 참조 카운트 방식의 메모리 관리
모든 객체는 참조 카운트가 0인 상태로 생성되고, 다른 객체가 참조할 때마다 1씩 증가
- (ex) 대입 연산자로 사용될 때, 함수의 인자로 전달될 때, 컨테이너 타입에 추가될 때
객체에 대한 참조가 끊길 때마다 1만큼 감소하다가 0이 되면 파이썬 인터프리터가 회수

## 17.3 순환 참조와 가비지 컬렉션
- 순환 참조: 참조 수는 1이상인데 사용자가 절대 접근할 수 없는 상황
- 메모리가 부족해지는 시점에 파이썬 인터프리터에 의해 자동으로 호출됨

## 17.4 weakref 모듈
객체를 참조하되 참조 카운트는 증가하지 않는 기능
'''
import weakref

import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"<{type(data)}>는 지원하지 않습니다.")

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs]) 
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
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
        x = self.inputs[0].data
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

    def backward(self, gy):
        return gy, gy


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
    # 17.4 weakref 모듈
    a = np.array([1, 2, 3])
    b = weakref.ref(a)

    print(b)
    print(b())

    a = None
    print(b)
    print(b())

    # 17.5 동작 확인
    for i in range(10):
        x = Variable(np.random.rand(10000))
        y = square(square(square(x)))
