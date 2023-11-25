import torch
import torch.nn as nn


class InputStem(nn.Module):
    def __init__(self):
        """
            Входной блок нейронной сети ResNet, содержит свертку 7x7 c количеством фильтров 64 и шагом 2, затем
            следует max-pooling 3x3 с шагом 2.
            
            TODO: инициализируйте слои входного блока
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        raise NotImplementedError


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1, down_sampling=False):
        """
            Остаточный блок, состоящий из 3 сверточных слоев (path A) и shortcut connection (path B).
            Может быть двух видов:
                1. Down sampling (только первый Bottleneck блок в Stage)
                2. Residual (последующие Bottleneck блоки в Stage)

            Path A:
                Cостоит из 3-x сверточных слоев (1x1, 3x3, 1x1), после каждого слоя применяется BatchNorm,
                после первого и второго слоев - ReLU. Количество фильтров для первого слоя - out_channels,
                для второго слоя - out_channels, для третьего слоя - out_channels * expansion.

            Path B:
                1. Down sampling: path B = Conv (1x1, stride) и  BatchNorm
                2. Residual: path B = nn.Identity

            Выход Bottleneck блока - path_A(inputs) + path_B(inputs)

            :param in_channels: int - количество фильтров во входном тензоре
            :param out_channels: int - количество фильтров в промежуточных слоях
            :param expansion: int = 4 - множитель на количество фильтров в выходном слое
            :param stride: int
            :param down_sampling: bool
            TODO: инициализируйте слои Bottleneck
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        raise NotImplementedError


class Stage(nn.Module):
    def __init__(self, nrof_blocks: int):
        """
            Последовательность Bottleneck блоков, первый блок Down sampling, остальные - Residual

            :param nrof_blocks: int - количество Bottleneck блоков
            TODO: инициализируйте слои, используя класс Bottleneck
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        raise NotImplementedError
