import torch
import torch.nn as nn
from models.blocks.resnet_blocks import InputStem, Stage


class ResNet50(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """ https://arxiv.org/pdf/1512.03385.pdf """
        super(ResNet50, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # TODO: инициализируйте слои модели, используя классы InputStem, Stage
        self.input_block = ...
        self.stages = ...

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # TODO: инициализируйте выходной слой модели
        self.linear = ...

        self.apply(self._init_weights)
        raise NotImplementedError

    def _init_weights(self, m):
        """
            Cверточные и полносвязные веса инициализируются согласно xavier_uniform
            Все bias инициализируются 0
            В слое batch normalization вектор gamma инициализируется 1, вектор beta – 0 (в базовой модели)

            # TODO: реализуйте этот метод
        """
        raise NotImplementedError

    def weight_decay_params(self):
        """
            Сбор параметров сети, для которых применяется (веса сверточных и полносвязных слоев)
            и не применяется L2-регуляризация (все остальные параметры, включая bias conv и linear)
            :return: wo_decay, w_decay

            # TODO: реализуйте этот метод
        """
        wo_decay, w_decay = [], []
        raise NotImplementedError

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight), channels = 3, height = weight = 224
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
       """
        raise NotImplementedError
