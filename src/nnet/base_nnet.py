import datetime
import typing
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Model, metrics, losses
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.types.core import Tensor

from src.nnet.freeze_callback import FreezeCallback
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
        self.build_model()
        self.model.summary()

    @abstractmethod
    def build_model(self):
        """
        モデルをビルド
        """

        raise NotImplementedError()

    def compile_model(self, loss: typing.Callable):
        """
        モデルをコンパイル
        """

        self.model.compile(
            optimizer="adam",
            loss=loss,
            metrics=[self.θ_metric, self.σ_metric]
        )

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
            results[:, 1] = results[:, 1] * (self.human_property.max_age - self.human_property.min_age)

        return results

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        """
        学習

        :param x_train: 入力データ（学習用）
        :param y_train: 正解ラベル（学習用）
        :param x_test: 入力データ（検証用）
        :param y_test: 正解ラベル（検証用）
        """

        if self.nnet_property.normalize:
            y_train = (y_train - self.human_property.min_age) \
                      / (self.human_property.max_age - self.human_property.min_age)
            y_test = (y_test - self.human_property.min_age) \
                     / (self.human_property.max_age - self.human_property.min_age)

        # チェックポイントを保存するコールバックを定義
        checkpoint_filename: str = f"{self.path_property.checkpoint_path}/{self.nnet_property.checkpoint_filename}"
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filename,
            verbose=1,
            save_weights_only=True
        )
        self.model.save_weights(checkpoint_filename.format(epoch=0))

        # ロギングするコールバックを定義
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        logging_callback = CSVLogger(
            filename=f"{self.path_property.log_path}/{self.logging_property.train_filename.format(timestamp=timestamp)}",
            separator=",",
            append=True,
        )

        callbacks: list[Callback] = [checkpoint_callback, logging_callback]
        if self.nnet_property.freeze:
            # バッチ毎にモデルの一部をフリーズするコールバックを定義
            freeze_callback = FreezeCallback()
            callbacks.append(freeze_callback)

        # 学習
        self.compile_model(self.second_loss)
        return self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=256,
            epochs=self.nnet_property.epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
        )

    def load_weights(self, filename):
        """
        学習済みモデルをロード
        """

        self.model.load_weights(filename)

    def first_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tensor:
        """
        損失関数（1番目）

        :param y_true: NNの出力
        :param y_pred: 正解ラベル
        :return: loss
        """

        # y: 正解の年齢
        # θ: 推定した年齢
        # σ: 推定した残差標準偏差
        y = y_true[:, 0]
        θ = y_pred[:, 0]

        return losses.mean_squared_error(y, θ)

    def second_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tensor:
        """
        損失関数（2番目）

        :param y_true: NNの出力
        :param y_pred: 正解ラベル
        :return: loss
        """

        # y: 正解の年齢
        # θ: 推定した年齢
        # σ: 推定した残差標準偏差
        y = y_true[:, 0]
        θ = y_pred[:, 0]
        ρ = y_pred[:, 1]

        # LaTeX: log(2 \pi \sigma^2) + \frac{(y - \theta)^2}{\sigma^2}
        return K.mean(
            ρ + (y - θ) ** 2 * K.exp(-ρ)
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
        ρ = y_pred[:, 1] + K.constant(K.epsilon())

        return K.stack([θ, ρ], 1)

    def θ_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tensor:
        """
        年齢θの評価関数

        :param y_true: NNの出力
        :param y_pred: 正解ラベル
        :return: θの評価関数
        """

        # y: 正解の年齢
        # θ: 推定した年齢
        # σ: 推定した残差標準偏差
        y = y_true[:, 0]
        θ = y_pred[:, 0]

        if self.nnet_property.normalize:
            max_age_tensor = K.constant(self.human_property.max_age)
            min_age_tensor = K.constant(self.human_property.min_age)

            y = y * (max_age_tensor - min_age_tensor) + min_age_tensor
            θ = θ * (max_age_tensor - min_age_tensor) + min_age_tensor

        return metrics.mean_absolute_error(y, θ)

    def σ_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tensor:
        """
        残差標準偏差σの評価関数

        :param y_true: NNの出力
        :param y_pred: 正解ラベル
        :return: σの評価関数
        """

        # y: 正解の年齢
        # θ: 推定した年齢
        # σ: 推定した残差標準偏差
        y = y_true[:, 0]
        θ = y_pred[:, 0]
        σ = y_pred[:, 1]

        if self.nnet_property.normalize:
            max_age_tensor = K.constant(self.human_property.max_age)
            min_age_tensor = K.constant(self.human_property.min_age)

            y = y * (max_age_tensor - min_age_tensor) + min_age_tensor
            θ = θ * (max_age_tensor - min_age_tensor) + min_age_tensor
            σ = σ * (max_age_tensor - min_age_tensor)

        return metrics.mean_absolute_error(K.abs(y - θ), σ)
