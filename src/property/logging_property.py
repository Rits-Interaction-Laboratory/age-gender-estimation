from .base_property import BaseProperty


class LoggingProperty(BaseProperty):
    """
    ロギングプロパティ
    """

    train_filename: str = ""
    """
    
    学習ログのファイル名
    """

    dataset_loader_filename: str = ""
    """
    データセット読み込みログのファイル名
    """
