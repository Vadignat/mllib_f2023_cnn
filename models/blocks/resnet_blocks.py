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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(3, 2, padding=1)

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        inputs = self.maxpool1(inputs)
        return inputs

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
        self.path_a = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1),
            nn.BatchNorm2d(out_channels*expansion)
        )

        if down_sampling:
            self.path_b = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * expansion)
            )
        else:
            self.path_b = nn.Identity()

        self.relu = nn.ReLU(inplace=True)


    def forward(self, inputs):
        # TODO: реализуйте forward pass
        residual = inputs

        inputs = self.path_a(inputs)
        inputs = inputs + self.path_b(residual)
        inputs = self.relu(inputs)

        return inputs


class Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nrof_blocks: int, expansion : int = 4, stride: int = 1):
        """
            Последовательность Bottleneck блоков, первый блок Down sampling, остальные - Residual

            :param nrof_blocks: int - количество Bottleneck блоков
            TODO: инициализируйте слои, используя класс Bottleneck
        """
        super().__init__()


        self.first_block = Bottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            down_sampling=True
        )

        self.blocks = nn.ModuleList([
            Bottleneck(
                in_channels=out_channels * expansion,
                out_channels=out_channels
            ) for _ in range(1, nrof_blocks)
        ])


    def forward(self, inputs):
        # TODO: реализуйте forward pass
        inputs = self.first_block(inputs)

        for block in self.blocks:
            inputs = block(inputs)

        return inputs
