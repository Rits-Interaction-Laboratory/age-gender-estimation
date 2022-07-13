import numpy as np

from src.nnet.base_nnet import BaseNNet
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

    def __init__(self, nnet: BaseNNet):
        self.nnet = nnet

    def train(self):
        """
        学習する
        """

        # データセットをロード
        humans = self.human_repository.select_all()
        humans_train, humans_test = self.human_repository.split_dataset(humans)
        x_train: np.ndarray = np.array([human.image for human in humans_train], dtype=np.float32)
        y_train: np.ndarray = np.array([human.age for human in humans_train], dtype=np.float32)
        x_test: np.ndarray = np.array([human.image for human in humans_test], dtype=np.float32)
        y_test: np.ndarray = np.array([human.age for human in humans_test], dtype=np.float32)

        # 学習
        self.nnet.train(x_train, y_train, x_test, y_test)
