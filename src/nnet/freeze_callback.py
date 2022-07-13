from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback


class FreezeCallback(Callback):
    """
    モデルをフリーズするコールバック
    """

    model: Model
    """
    モデル
    """

    def on_train_batch_begin(self, batch: int, logs: dict = None):
        """
        訓練データのバッチ開始時
        """

        number_of_layers: int = len(self.model.layers)

        # レイヤーを分割する境界
        layer_split_index: int = 3

        # バッチ単位でモデルのフリーズ箇所をスイッチ
        if batch % 2 == 0:
            # 特徴抽出部をフリーズ
            for i in range(number_of_layers - layer_split_index):
                self.model.layers[number_of_layers - i - 1].trainable = False
        else:
            # 出力部をフリーズ
            for i in range(layer_split_index):
                self.model.layers[i].trainable = False

    def on_train_batch_end(self, batch, logs=None):
        """
        訓練データのバッチ終了時
        """

        number_of_layers: int = len(self.model.layers)

        for i in range(number_of_layers):
            self.model.layers[i].trainable = True
