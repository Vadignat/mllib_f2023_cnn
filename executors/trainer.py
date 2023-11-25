# TODO: Реализуйте класс для обучения моделей, минимальный набор функций:
#  1. Подготовка обучающих и тестовых данных
#  2. Подготовка модели, оптимайзера, целевой функции
#  3. Обучение модели на обучающих данных
#  4. Эвалюэйшен модели на тестовых данных, для оценки точности можно рассмотреть accuracy, balanced accuracy
#  5. Сохранение и загрузка весов модели
#  6. Добавить возможность обучать на gpu
#  За основу данного класса можно взять https://github.com/pkhanzhina/mllib_f2023_mlp/blob/master/executors/mlp_trainer.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.oxford_pet_dataset import OxfordIIITPet
from logs.Logger import Logger
from models.vgg16 import VGG16
from models.resnet50 import ResNet50
from utils.metrics import accuracy, balanced_accuracy
from utils.visualization import show_batch
from utils.utils import set_seed


class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)

        self.cfg = cfg
