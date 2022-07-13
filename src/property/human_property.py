from .base_property import BaseProperty


class HumanProperty(BaseProperty):
    """
    人間プロパティ
    """

    shape: list[int] = []
    """
    画像の形状
    """

    min_age: int = 0
    """
    最小年齢
    """

    max_age: int = 0
    """
    最大年齢
    """
