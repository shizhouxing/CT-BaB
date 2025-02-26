import torch.nn as nn


def get_activation(activation):
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'leaky_relu':
        activation = nn.LeakyReLU()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        raise NotImplementedError
    return activation


class FFNet(nn.Module):
    def __init__(self, in_dim, out_dim, depth, width, bias=True, last_bias=True, activation='relu'):
        super().__init__()
        self.depth = depth
        if isinstance(width, int):
            width = [width]
        if len(width) == 1:
            width = width * (depth - 1)
        self.width = width
        if not bias:
            last_bias = False
        activation = get_activation(activation)
        if depth == 1:
            layers = [nn.Linear(in_dim, out_dim, bias=last_bias)]
        else:
            layers = [nn.Linear(in_dim, width[0], bias=bias)]
            layers.append(activation)
            for i in range(depth - 2):
                layers.append(nn.Linear(width[i], width[i + 1], bias=bias))
                layers.append(activation)
            layers.append(nn.Linear(width[-1], out_dim, bias=last_bias))
        self.net = nn.Sequential(*layers)
        self.project_layer = self.net[len(layers) - 1]

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim, depth, width: int, bias=True, activation='relu'):
        super().__init__()
        self.depth = depth
        self.width = width
        self.activation = get_activation(activation)

        assert depth % 2 == 0 and depth >= 4
        self.layers = nn.ModuleList([nn.Linear(in_dim, width)])
        for i in range(depth // 2 - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, out_dim))
        self.project_layer = self.layers[-1]

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers) // 2 - 1):
            z = self.layers[i * 2](self.activation(self.layers[i * 2 - 1](x)))
            x = self.activation(x + z)
        x = self.layers[-1](x)
        return x
