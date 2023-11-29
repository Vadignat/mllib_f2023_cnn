import torch
import torch.nn as nn

from models.blocks.vgg16_blocks import conv_block, classifier_block


class VGG16(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """https://arxiv.org/pdf/1409.1556.pdf"""
        super(VGG16, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # TODO: инициализируйте сверточные слои модели, используя функцию conv_block
        self.conv1 = conv_block(in_channels=[3, 64], out_channels=[64, 64])
        self.conv2 = conv_block(in_channels=[64, 128], out_channels=[128, 128])
        self.conv3 = conv_block(in_channels=[128, 256, 256], out_channels=[256, 256, 256])
        self.conv4 = conv_block(in_channels=[256, 512, 512], out_channels=[512, 512, 512])
        self.conv5 = conv_block(in_channels=[512, 512, 512], out_channels=[512, 512, 512])

        # TODO: инициализируйте полносвязные слои модели, используя функцию classifier_block
        #  (последний слой инициализируется отдельно)
        self.linears = classifier_block(in_features=[25088, 4096], out_features=[4096, 4096])

        # TODO: инициализируйте последний полносвязный слой для классификации с помощью
        #  nn.Linear(in_features=4096, out_features=nrof_classes)
        self.classifier = nn.Linear(in_features=4096, out_features=nrof_classes)


    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight)
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
        """
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        inputs = self.conv4(inputs)
        inputs = self.conv5(inputs)


        inputs = inputs.view(inputs.size(0), -1)

        inputs = self.linears(inputs)
        inputs = self.classifier(inputs)

        return inputs
