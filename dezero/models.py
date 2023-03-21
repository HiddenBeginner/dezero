import dezero.functions as F
import dezero.layers as L
from dezero import Layer, utils


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, hidden_dims, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = L.Linear(hidden_dim)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            return self.layers[-1](x)
