from src.repository.human_repository import HumanRepository


class TrainUseCase:
    """
    学習のユースケース
    """

    human_repository: HumanRepository = HumanRepository()
    """
    人間リポジトリ
    """

    def train(self):
        """
        学習する
        """

        # TODO: 学習する
