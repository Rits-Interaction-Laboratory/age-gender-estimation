from .base_property import BaseProperty


class LoggingProperty(BaseProperty):
    """
    ロギングプロパティ
    """

    filename: str
    """
    ログファイル名
    """
