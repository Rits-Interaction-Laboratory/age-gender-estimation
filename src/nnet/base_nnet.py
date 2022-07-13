import time
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.types.core import Tensor

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

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tensor:
        """
        損失関数

        :param y_true: NNの出力
        :param y_pred: 正解ラベル
        :return: loss
        """

        # y: 正解の年齢
        # θ: 推定した年齢
        # σ: 推定した残差標準偏差
        y = y_true[:, 0]
        θ = y_pred[:, 0]
        σ = y_pred[:, 1]

        # LaTeX: log(2 \pi \sigma^2) + \frac{(y - \theta)^2}{\sigma^2}
        return K.mean(
            K.log(2 * np.pi * (σ ** 2)) + ((y - θ) ** 2) / (σ ** 2 + K.constant(K.epsilon()))
        )

    def output_activation(self, y_pred: np.ndarray) -> Tensor:
        """
        出力層の活性化関数

        :param y_pred: NNの出力
        :return: 活性化関数を通した出力
        """

        # θ: 推定した年齢
        # σ: 推定した残差標準偏差
        θ = y_pred[:, 0]
        σ = y_pred[:, 1]

        if self.nnet_property.normalize:
            θ = K.sigmoid(θ)
            σ = K.sigmoid(σ)

        return K.stack([θ, σ], 1)
