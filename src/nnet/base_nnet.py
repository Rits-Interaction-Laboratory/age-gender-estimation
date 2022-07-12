import time
from abc import ABCMeta, abstractmethod

import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger

from src.property.human_property import HumanProperty
from src.property.logging_property import LoggingProperty
from src.property.nnet_property import NNetProperty
from src.property.path_property import PathProperty


class BaseNNet(metaclass=ABCMeta):
    """
    ニューラルネットワークの基底クラス
    """

    path_property: PathProperty = PathProperty()
    """
    PATHプロパティ
    """

    human_property: HumanProperty = HumanProperty()
    """
    人間プロパティ
    """

    nnet_property: NNetProperty = NNetProperty()
    """
    ニューラルネットプロパティ
    """

    logging_property: LoggingProperty = LoggingProperty()
    """
    ロギングプロパティ
    """

    model: Model
    """
    モデル
    """

    def __init__(self):
        pass

    @abstractmethod
    def build_model(self):
        """
        モデルをビルド
        """

        raise NotImplementedError()

    def compile_model(self):
        """
        モデルをコンパイル
        """

        self.model.compile(optimizer="adam")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推論

        :param x: 入力データリスト
        :return: 推論結果リスト
        """

        results = self.model.predict(x)

        if self.nnet_property.normalize:
            results[:, 0] = results[:, 0] \
                            * (self.human_property.max_age - self.human_property.min_age) + self.human_property.min_age
            results[:, 1] = results[:, 1] \
                            * (self.human_property.max_age - self.human_property.min_age)

        return results

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        """
        学習

        :param x_train: 入力データ（学習用）
        :param y_train: 正解ラベル（学習用）
        :param x_test: 入力データ（検証用）
        :param y_test: 正解ラベル（検証用）
        """

        # チェックポイントを保存するコールバックを定義
        checkpoint_filename: str = f"{self.path_property.checkpoint_path}/${self.nnet_property.checkpoint_filename}"
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filename,
            verbose=1,
            save_weights_only=True
        )
        self.model.save_weights(checkpoint_filename.format(epoch=0))

        # ロギングするコールバックを定義
        timestamp = int(time.time())
        logging_callback = CSVLogger(self.logging_property.filename.format(timestamp=timestamp))

        # 学習
        self.model.fit(
            x_train,
            x_test,
            epochs=self.nnet_property.epochs,
            batch_size=self.nnet_property.batch_size,
            validation_data=(y_train, y_test),
            callbacks=[checkpoint_callback, logging_callback],
        )

    def load_weights(self):
        """
        学習済みモデルをロード
        """

        filename: str = f"{self.path_property.checkpoint_path}/{self.nnet_property.weights_filename}"
        self.model.load_weights(filename)
