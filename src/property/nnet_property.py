from .base_property import BaseProperty


class NNetProperty(BaseProperty):
    """
    ニューラルネットプロパティ
    """

    epochs: int
    """
    エポック数
    """

    batch_size: int
    """
    バッチサイズ
    """

    usage_rate: float
    """
    使用するデータセットの割合
    """

    validation_split_rate: float
    """
    検証用データの割合
    """

    normalize: bool
    """
    正規化するか
    """

    freeze: bool
    """
    バッチ毎にモデルの一部をフリーズするか
    """

    weights_filename: str
    """
    学習済みモデルのファイル名
    """

    checkpoint_filename: str
    """
    チェックポイントのファイル名
    """
