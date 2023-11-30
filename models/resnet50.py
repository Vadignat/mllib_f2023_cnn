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
        self.input_block = InputStem()
        #self.stages =
        self.stage1 = Stage(64, 64, self.cfg.nrof_blocks[0])
        self.stage2 = Stage(256, 128, self.cfg.nrof_blocks[1])
        self.stage3 = Stage(512, 64, self.cfg.nrof_blocks[2])
        self.stage4 = Stage(512, 512, self.cfg.nrof_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # TODO: инициализируйте выходной слой модели
        self.linear = nn.Linear(2048, self.nrof_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
            Cверточные и полносвязные веса инициализируются согласно xavier_uniform
            Все bias инициализируются 0
            В слое batch normalization вектор gamma инициализируется 1, вектор beta – 0 (в базовой модели)

            # TODO: реализуйте этот метод
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def weight_decay_params(self):
        """
            Сбор параметров сети, для которых применяется (веса сверточных и полносвязных слоев)
            и не применяется L2-регуляризация (все остальные параметры, включая bias conv и linear)
            :return: wo_decay, w_decay

            # TODO: реализуйте этот метод
        """
        wo_decay, w_decay = [], []
        for name, param in self.named_parameters():
            if 'weight' in name and 'BatchNorm' not in name:
                w_decay.append(param)
            else:
                wo_decay.append(param)
        return wo_decay, w_decay

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight), channels = 3, height = weight = 224
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
       """
        inputs = self.input_block(inputs)
        inputs = self.stage1(inputs)
        inputs = self.stage2(inputs)
        inputs = self.stage3(inputs)
        inputs = self.stage4(inputs)
        inputs = self.avg_pool(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = self.linear(inputs)

        return inputs
