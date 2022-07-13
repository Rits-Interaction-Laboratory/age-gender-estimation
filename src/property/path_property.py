from .base_property import BaseProperty


class PathProperty(BaseProperty):
    """
    PATHプロパティ
    """

    data_path: str = ""
    """
    データを保存するパス
    """

    checkpoint_path: str = ""
    """
    チェックポイントを保存するパス
    """

    log_path: str = ""
    """
    ログを保存するパス
    """

    heatmap_path: str = ""
    """
    ヒートマップを保存するパス
    """
