from enum import Enum


class Task(Enum):
    ERM = 'erm'
    VAE = 'vae'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'