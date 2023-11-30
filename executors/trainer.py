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

from my_datasets.utils.prepare_transforms import prepare_transforms
from my_datasets.oxford_pet_dataset import OxfordIIITPet
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
        self.device = cfg.device

        # TODO: настройте логирование с помощью класса Logger
        #  (пример: https://github.com/KamilyaKharisova/mllib_f2023/blob/master/logginig_example.py)

        # TODO: залогируйте используемые гиперпараметры в neptune.ai через метод log_hyperparameters
        self.neptune_logger = Logger(cfg.env_path, cfg.project_name)
        self.neptune_logger.log_hyperparameters(params={
        'learning_rate': cfg.lr,
        'batch_size': cfg.batch_size,
        'optimizer': cfg.optimizer_name
    })

        self.__prepare_data(self.cfg.dataset_cfg)
        self.__prepare_model(self.cfg.model_cfg, self.cfg.dataset_cfg.nrof_classes)

    def __prepare_data(self, dataset_cfg):
        """ Подготовка обучающих и тестовых данных """
        self.train_dataset = OxfordIIITPet(dataset_cfg, 'train',
                                    transform=prepare_transforms(dataset_cfg.transforms['train']))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True)

        self.test_dataset = OxfordIIITPet(dataset_cfg, 'test', transform=prepare_transforms(dataset_cfg.transforms['test']))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True)

    def __prepare_model(self, model_cfg, nrof_classes):
        """ Подготовка нейронной сети"""
        model_class = globals().get(self.cfg.model_name)
        self.model = model_class(model_cfg, nrof_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        nrof_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'number of trainable parameters: {nrof_params}')

        # TODO: инициализируйте оптимайзер через getattr(torch.optim, self.cfg.optimizer_name)
        self.optimizer = getattr(torch.optim, self.cfg.optimizer_name)(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum = self.cfg.momentum,
            weight_decay = self.cfg.weight_decay
        )

    def save_model(self, filename):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            TODO: реализовать сохранение модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        if not os.path.exists(self.cfg.exp_dir):
            os.makedirs(self.cfg.exp_dir)


        path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename):
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            TODO: реализовать выгрузку весов модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        self.model.load_state_dict(torch.load(path))

    def make_step(self, batch, update_model=True):
        """
            Этот метод выполняет один шаг обучения, включая forward pass, вычисление целевой функции,
            backward pass и обновление весов модели (если update_model=True).

            :param batch: dict of data with keys ["image", "label"]
            :param update_model: bool - если True, необходимо сделать backward pass и обновить веса модели
            :return: значение функции потерь, выход модели
            # TODO: реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
        labels = labels.long()


        logits = self.model(inputs)
        loss = self.criterion(logits, labels)

        if update_model:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), logits


    def train_epoch(self, *args, **kwargs):
        """
            Обучение модели на self.train_dataloader в течение одной эпохи. Метод проходит через все обучающие данные и
            вызывает метод self.make_step() на каждом шаге.

            TODO: реализуйте функцию обучения с использованием метода self.make_step(batch, update_model=True),
                залогируйте на каждом шаге значение целевой функции и accuracy на batch
        """
        self.model.train()


        for batch_idx, batch in enumerate(self.train_dataloader):
            show_batch(batch['image'].to(self.device))
            loss, logits = self.make_step(batch, update_model=True)


            _, predicted_labels = torch.max(logits, 1)
            accuracy_value = accuracy(predicted_labels, batch['label'].to(self.device))
            balanced_accuracy_value = balanced_accuracy(predicted_labels, batch['label'].to(self.device), self.cfg.dataset_cfg.nrof_classes)
            self.neptune_logger.save_param(
                'train',
                ['target_function_value', 'accuracy', 'balanced_accuracy', 'learning_rate'],
                [loss, accuracy_value, balanced_accuracy_value, self.optimizer.param_groups[0]['lr']]
            )



    def evaluate(self, *args, **kwargs):
        """
            Метод используется для проверки производительности модели на обучающих/тестовых данных. Сначала модель
            переводится в режим оценки (model.eval()), затем данные последовательно подаются на вход модели, по
            полученным выходам вычисляются метрики производительности, такие как значение целевой функции, accuracy

            TODO: реализуйте функцию оценки с использованием метода self.make_step(batch, update_model=False),
                залогируйте значения целевой функции и accuracy, постройте confusion_matrix
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):

                loss, logits = self.make_step(batch, update_model=False)

                _, predicted_labels = torch.max(logits.float(), 1)
                total_loss += loss

                all_predictions.extend(predicted_labels.tolist())
                all_labels.extend(batch['label'].tolist())

        preds = torch.tensor(all_predictions)
        labels = torch.tensor(all_labels)

        accuracy_value = accuracy(preds, labels)
        balanced_accuracy_value = balanced_accuracy(preds, labels, self.cfg.dataset_cfg.nrof_classes)
        self.neptune_logger.save_param(
            'train/test',
            ['target_function_value', 'accuracy', 'balanced_accuracy'],
            [total_loss, accuracy_value, balanced_accuracy_value]
        )

        return accuracy_value

    def fit(self, num_epochs : int = 10):
        """
            Основной цикл обучения модели. Данная функция должна содержать один цикл на заданное количество эпох.
            На каждой эпохе сначала происходит обучение модели на обучающих данных с помощью метода self.train_epoch(),
            а затем оценка производительности модели на тестовых данных с помощью метода self.evaluate()

            # TODO: реализуйте основной цикл обучения модели, сохраните веса модели с лучшим значением accuracy на
                тестовой выборке
        """
        best_accuracy = 0.0


        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")

            self.train_epoch()

            accuracy = self.evaluate()

            print('[{:d}]: accuracy {:.4f}'.format(epoch, accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model("best_model")

    def overfitting_on_batch(self, max_step=100):
        """
            Оверфиттинг на одном батче. Эта функция может быть полезна для отладки и оценки способности вашей
            модели обучаться и обновлять свои веса в ответ на полученные данные.
        """
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 10 == 0:
                _, predicted_labels = torch.max(output, 1)
                acc = accuracy(predicted_labels, batch['label'])
                print('[{:d}]: loss - {:.4f}, {:.4f}'.format(step + 1, loss, acc))