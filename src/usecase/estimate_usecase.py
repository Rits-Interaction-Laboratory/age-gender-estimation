import numpy as np
from matplotlib import pyplot as plt

from src.nnet.base_nnet import BaseNNet
from src.property.path_property import PathProperty
from src.repository.human_repository import HumanRepository


class EstimateUseCase:
    """
    推定のユースケース
    """

    path_property: PathProperty = PathProperty()
    """
    PATHプロパティ
    """

    nnet: BaseNNet
    """
    ニューラルネット
    """

    human_repository: HumanRepository = HumanRepository()
    """
    人間リポジトリ
    """

    def __init__(self, nnet: BaseNNet, weights_filename: str):
        self.nnet = nnet
        self.nnet.load_weights(weights_filename)

    def estimate_age(self):
        """
        年齢を推定
        """

        # データセットを読み込む
        humans = self.human_repository.select_all()
        humans_train, humans_test = self.human_repository.split_dataset(humans)
        x_train: np.ndarray = np.array([human.image for human in humans_train], dtype=np.float32)
        x_test: np.ndarray = np.array([human.image for human in humans_test], dtype=np.float32)

        # 年齢を推定
        results_train = self.nnet.predict(x_train)
        results_test = self.nnet.predict(x_test)

        θ_pred_list_train: list[float] = []
        θ_true_list_train: list[float] = []
        σ_pred_list_train: list[float] = []
        σ_true_list_train: list[float] = []
        θ_pred_list_test: list[float] = []
        θ_true_list_test: list[float] = []
        σ_pred_list_test: list[float] = []
        σ_true_list_test: list[float] = []

        for i in range(len(results_train)):
            human = humans_train[i]
            θ, σ = results_train[i]
            θ_pred_list_train.append(θ)
            σ_pred_list_train.append(σ)
            θ_true_list_train.append(human.age)
            σ_true_list_train.append(abs(human.age - θ))

        for i in range(len(results_test)):
            human = humans_test[i]
            θ, σ = results_test[i]
            θ_pred_list_test.append(θ)
            σ_pred_list_test.append(σ)
            θ_true_list_test.append(human.age)
            σ_true_list_test.append(abs(human.age - θ))

        # θのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        plt.hist2d(θ_pred_list_train, θ_true_list_train, bins=90, range=[(0, 90), (0, 90)], cmin=1)
        plt.colorbar()
        plt.xlabel("θ")
        plt.ylabel("Age")
        plt.xlim(0, 90)
        plt.ylim(0, 90)
        plt.savefig(f"{self.path_property.heatmap_path}/θ_train.png")

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        plt.hist2d(θ_pred_list_test, θ_true_list_test, bins=90, range=[(0, 90), (0, 90)], cmin=1)
        plt.colorbar()
        plt.xlabel("θ")
        plt.ylabel("Age")
        plt.xlim(0, 90)
        plt.ylim(0, 90)
        plt.savefig(f"{self.path_property.heatmap_path}/θ_test.png")

        # 残差標準偏差σのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_list_train, σ_true_list_train, bins=80, range=[(0, 50), (0, 50)], cmin=1)
        plt.colorbar()
        plt.xlabel("σ")
        plt.ylabel("|y - θ|")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_train.png")

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_list_test, σ_true_list_test, bins=80, range=[(0, 50), (0, 50)], cmin=1)
        plt.colorbar()
        plt.xlabel("σ")
        plt.ylabel("|y - θ|")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_test.png")
