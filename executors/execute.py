from easydict import EasyDict
import os
from executors.trainer import Trainer

# конфиг для датасета
ROOT_DIR = r"C:\Users\vadim\Downloads\archive"

dataset_cfg = EasyDict()

dataset_cfg.path = os.path.join(ROOT_DIR, 'oxford-iiit-pet')
dataset_cfg.nrof_classes = 37

dataset_cfg.annotation_filenames = {
    'train': 'trainval.txt',
    'test': 'test.txt'
}

dataset_cfg.transforms = EasyDict()
dataset_cfg.transforms.train = [
    ('RandomResizedCrop', ((224, 224),)),
    ('ToTensor', ()),
    ('RandomHorizontalFlip', ()),
    ('ColorJitter', (0.4, 0.4, 0.4)),
    ('Normalize', ([0.485, 0.456, 0.405], [0.229, 0.224, 0.225]))
]
dataset_cfg.transforms.test = [
    ('Resize', ((256, 256),)),
    ('CenterCrop', ((224, 224),)),
    ('ToTensor', ()),
    ('Normalize', ([0.485, 0.456, 0.405], [0.229, 0.224, 0.225]))
]


# конфиг для модели
model_cfg = EasyDict()

# конфиг для обучения
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_cfg = EasyDict()
train_cfg.seed = 0

train_cfg.batch_size = 64
train_cfg.lr = 1e-3

train_cfg.model_name = 'VGG16'  # ['VGG16', 'ResNet50']
train_cfg.optimizer_name = 'Adam'  # ['SGD', 'Adam']

train_cfg.device = 'cpu'  # ['cpu', 'cuda']
train_cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_name')

train_cfg.env_path = os.path.join(ROOT_DIR,'.env')  # Путь до файла .env где будет храниться api_token.
train_cfg.project_name = 'linear-regression'

train_cfg.dataset_cfg = dataset_cfg
train_cfg.model_cfg = model_cfg

# инициализация трейнера
trainer = Trainer(train_cfg)

# обучение
trainer.overfitting_on_batch(max_step=10)
