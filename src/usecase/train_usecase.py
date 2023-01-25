import random

import numpy as np

from src.nnet.base_nnet import BaseNNet
from src.property.human_property import HumanProperty
from src.repository.human_repository import HumanRepository


class TrainUseCase:
    """
    学習のユースケース
    """

    nnet: BaseNNet
    """
    ニューラルネット
    """

    human_repository: HumanRepository = HumanRepository()
    """
    人間リポジトリ
    """

    human_property: HumanProperty = HumanProperty()
    """
    人間プロパティ
    """

    def __init__(self, nnet: BaseNNet):
        self.nnet = nnet

    def handle(self):
        """
        UseCase Handler
        """

        # データセットをロード
        humans_train = self.human_repository.select_train()
        humans_test = self.human_repository.select_test()

        random.shuffle(humans_train)
        random.shuffle(humans_test)

        x_train: np.ndarray = np.array([human.image for human in humans_train], dtype=np.float32)
        y_train: np.ndarray = np.array([human.age for human in humans_train], dtype=np.float32)
        x_test: np.ndarray = np.array([human.image for human in humans_test], dtype=np.float32)
        y_test: np.ndarray = np.array([human.age for human in humans_test], dtype=np.float32)

        # 学習
        return self.nnet.train(x_train, y_train, x_test, y_test)
