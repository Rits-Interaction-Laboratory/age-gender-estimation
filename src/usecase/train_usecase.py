import random

import numpy as np

from src.model.human_model import HumanModel
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
        humans_train_all = self.human_repository.select_train()
        humans_test = self.human_repository.select_test()

        # 各年齢の画像枚数を制限
        max_size: int = 200
        humans_train: list[HumanModel] = []
        for i in range(1, self.human_property.max_age + 1):
            humans = list(filter(lambda human: human.age == i, humans_train_all))
            random.shuffle(humans)
            humans_train.extend(humans[:max_size])

        random.shuffle(humans_train)
        random.shuffle(humans_test)

        x_train: np.ndarray = np.array([human.image for human in humans_train], dtype=np.float32)
        y_train: np.ndarray = np.array([human.age for human in humans_train], dtype=np.float32)
        x_test: np.ndarray = np.array([human.image for human in humans_test], dtype=np.float32)
        y_test: np.ndarray = np.array([human.age for human in humans_test], dtype=np.float32)

        # 学習
        self.nnet.train(x_train, y_train, x_test, y_test)
