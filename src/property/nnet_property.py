from .base_property import BaseProperty


class NNetProperty(BaseProperty):
    """
    ニューラルネットプロパティ
    """

    epochs: int = 0
    """
    エポック数
    """

    batch_size: int = 0
    """
    バッチサイズ
    """

    usage_rate: float = 0.0
    """
    使用するデータセットの割合
    """

    validation_split_rate: float = 0.0
    """
    検証用データの割合
    """

    normalize: bool = True
    """
    正規化するか
    """

    freeze: bool = True
    """
    バッチ毎にモデルの一部をフリーズするか
    """

    weights_filename: str = ""
    """
    学習済みモデルのファイル名
    """

    checkpoint_filename: str = ""
    """
    チェックポイントのファイル名
    """
