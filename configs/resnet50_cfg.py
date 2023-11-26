from easydict import EasyDict

from utils.enums import ModelName

cfg = EasyDict()

cfg.model_name = ModelName.ResNet50
cfg.nrof_blocks = [3, 4, 6, 3]
