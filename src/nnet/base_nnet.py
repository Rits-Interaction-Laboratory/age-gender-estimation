from abc import ABCMeta, abstractmethod

import numpy as np


class BaseNNet(metaclass=ABCMeta):
    """
    ニューラルネットワークの基底クラス
    """

    def __init__(self):
        pass

    @abstractmethod
    def build_model(self):
        """
        モデルをビルド
        """

        raise NotImplementedError()

    @abstractmethod
    def compile_model(self):
        """
        モデルをコンパイル
        """

        raise NotImplementedError()

    @abstractmethod
    def load_weights(self):
        """
        学習済みモデルを読み込み
        """

        raise NotImplementedError()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        """
        学習

        :param x_train: 入力データ（学習用）
        :param y_train: 正解ラベル（学習用）
        :param x_test: 入力データ（検証用）
        :param y_test: 正解ラベル（検証用）
        """

        # TODO: 学習する
        pass
