import yaml


class ApplicationProperty:
    """
    アプリケーションプロパティ
    """

    DATA_PATH: str = ""
    """
    データを保存するパス
    """

    CHECKPOINT_PATH: str = ""
    """
    チェックポイントを保存するパス
    """

    def __init__(self):
        with open("resources/application.yml", "r") as file:
            properties = yaml.safe_load(file)
            self.DATA_PATH = properties["path"]["data-path"]
            self.CHECKPOINT_PATH = properties["path"]["checkpoint-path"]
