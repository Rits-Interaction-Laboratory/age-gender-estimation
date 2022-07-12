from .base_property import BaseProperty


class PathProperty(BaseProperty):
    """
    PATHプロパティ
    """

    data_path: str
    """
    データを保存するパス
    """

    checkpoint_path: str
    """
    チェックポイントを保存するパス
    """
