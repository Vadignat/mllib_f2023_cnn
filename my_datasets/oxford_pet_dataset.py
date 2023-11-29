import os
import torch
from PIL import Image
from torch.utils.data import Dataset

from configs.oxford_pet_cfg import ROOT_DIR
from utils.visualization import show_batch


class OxfordIIITPet(Dataset):
    def __init__(self, cfg, dataset_type: str, transform=None):
        """
            https://www.robots.ox.ac.uk/~vgg/data/pets/
            https://www.kaggle.com/datasets/polinakhanzhina/oxford-iiit-pet
        """
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.transforms = transform

        self.paths, self.labels, self.categories = [], [], []

        self.prepare_data()

    def prepare_data(self):
        """
            Функция для подготовки данных, необходимо заполнить следующие списки в соответствии с self.dataset_type:
                - self.paths - пути до изображений
                - self.labels - лейбл изображений
                - self.categories - категория (cat или dog)

            Необходимо считать и распарсить файл аннотации
        """
        with open(os.path.join(self.cfg.path, 'annotations', self.cfg.annotation_filenames[self.dataset_type]), 'r') as f:
            lines = f.read()
        for line in lines.split('\n'):
            filename, label, category, _ = line.split()

            filename = f'{filename}.jpg'
            label = int(label) - 1
            category = int(category) - 1

            # TODO: реализуйте сохранение filename, label, category в соответствующие атрибуты класса
            path = os.path.join(ROOT_DIR, filename)
            self.paths.append(path)
            self.labels.append(label)
            self.categories.append(category)

    def __len__(self):
        """
            Функция __len__ возвращает количество элементов в наборе данных.
            TODO: Реализуйте этот метод
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
            Функция __getitem__ возвращает элемент из набора данных по заданному индексу idx.

            :param idx: int - представляет индекс элемента, к которому вы пытаетесь получить доступ из набора данных
            :return: dict - словарь с тремя ключами: "image", "label" и "category". Ключ "image" соответствует
            изображению, "label" соответствует метке этого изображения и "category" - категории изображения.
        """
        # TODO: Реализуйте считывание изображения через Image.open(path).convert("RGB") и применение self.transforms к
        #  этому изображению, верните словарь с ключами "image", "label", "category"

        image = Image.open(self.paths[idx]).convert("RGB")

        show_batch(image)

        if self.transforms is not None:
            image = self.transforms(image)
            show_batch(image)

        return {
            "image": image,
            "label": self.labels[idx],
            "category": self.categories[idx]
        }
