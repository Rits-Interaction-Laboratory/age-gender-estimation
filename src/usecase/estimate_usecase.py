import csv

import numpy as np
from matplotlib import pyplot as plt

from src.nnet.base_nnet import BaseNNet
from src.property.human_property import HumanProperty
from src.property.logging_property import LoggingProperty
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

    human_property: HumanProperty = HumanProperty()
    """
    人間プロパティ
    """

    log_property: LoggingProperty = LoggingProperty()
    """
    ロギングプロパティ
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

    def handle(self):
        """
        UseCase Handler
        """

        # データセットを読み込む
        humans_train = self.human_repository.select_train()
        humans_test = self.human_repository.select_test()
        x_train: np.ndarray = np.array([human.image for human in humans_train], dtype=np.float32)
        x_test: np.ndarray = np.array([human.image for human in humans_test], dtype=np.float32)

        plt.figure()
        plt.hist([human.age for human in humans_train], bins=self.human_property.max_age)
        plt.xlim(0, self.human_property.max_age)
        plt.xlabel(r"$x$")
        plt.savefig("./heatmap/histogram_train.png")

        plt.figure()
        plt.hist([human.age for human in humans_test], bins=self.human_property.max_age)
        plt.xlim(0, self.human_property.max_age)
        plt.xlabel(r"$x$")
        plt.savefig("./heatmap/histogram_test.png")

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
            θ = results_train[i][0]
            σ = np.sqrt(np.exp(results_train[i][1]))
            θ_pred_list_train.append(θ)
            σ_pred_list_train.append(σ)
            θ_true_list_train.append(human.age)
            σ_true_list_train.append(abs(human.age - θ))

        for i in range(len(results_test)):
            human = humans_test[i]
            θ = results_test[i][0]
            σ = np.sqrt(np.exp(results_test[i][1]))
            θ_pred_list_test.append(θ)
            σ_pred_list_test.append(σ)
            θ_true_list_test.append(human.age)
            σ_true_list_test.append(abs(human.age - θ))

        # 推定結果をログ出力
        with open(f"{self.path_property.heatmap_path}/{self.log_property.estimate_train_filename}", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "age", "θ", "σ", "filename"])
            for i in range(len(results_train)):
                human = humans_train[i]
                writer.writerow(["train", human.age, θ_pred_list_train[i], σ_pred_list_train[i], human.filename])
        with open(f"{self.path_property.heatmap_path}/{self.log_property.estimate_test_filename}", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "age", "θ", "σ", "filename"])
            for i in range(len(results_test)):
                human = humans_test[i]
                writer.writerow(["test", human.age, θ_pred_list_test[i], σ_pred_list_test[i], human.filename])

        # θのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        plt.hist2d(θ_pred_list_train, θ_true_list_train, bins=self.human_property.max_age,
                   range=[(0, self.human_property.max_age), (0, self.human_property.max_age)], cmin=1)
        plt.colorbar()
        plt.xlabel(r"$\hat{x}$")
        plt.ylabel(r"$x$")
        plt.xlim(0, self.human_property.max_age)
        plt.ylim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/θ_train.png")

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        plt.hist2d(θ_pred_list_test, θ_true_list_test, bins=self.human_property.max_age,
                   range=[(0, self.human_property.max_age), (0, self.human_property.max_age)], cmin=1)
        plt.colorbar()
        plt.xlabel(r"$\hat{x}$")
        plt.ylabel(r"$x$")
        plt.xlim(0, self.human_property.max_age)
        plt.ylim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/θ_test.png")

        # 残差標準偏差σのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_list_train, σ_true_list_train, bins=80, range=[(0, 50), (0, 50)], cmin=1)
        plt.colorbar()
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel(r"$|x - \hat{x}|$")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_train.png")

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_list_test, σ_true_list_test, bins=80, range=[(0, 50), (0, 50)], cmin=1)
        plt.colorbar()
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel(r"$|x - \hat{x}|$")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_test.png")

        # 残差標準偏差のヒートマップを作成
        σ_pred_standard_deviation_list_train: list[float] = []
        standard_deviation_list_train: list[float] = []
        σ_pred_standard_deviation_list_test: list[float] = []
        standard_deviation_list_test: list[float] = []
        σ_error_bar_list_train: list[float] = []
        σ_error_bar_x_list_train: list[float] = []
        σ_error_bar_list_test: list[float] = []
        σ_error_bar_x_list_test: list[float] = []

        for i in range(self.human_property.max_age):
            train_list: list[float] = []
            sum_train: float = 0.0
            cnt_train: int = 0
            test_list: list[float] = []
            sum_test: float = 0.0
            cnt_test: int = 0

            for j, σ in enumerate(σ_pred_list_train):
                if int(σ) == i:
                    train_list.append(σ_true_list_train[j])
                    sum_train += σ_true_list_train[j] ** 2
                    cnt_train += 1

            for j, σ in enumerate(σ_pred_list_test):
                if int(σ) == i:
                    test_list.append(σ_true_list_test[j])
                    sum_test += σ_true_list_test[j] ** 2
                    cnt_test += 1

            if cnt_train >= 1:
                for _ in range(cnt_train):
                    σ_pred_standard_deviation_list_train.append(i)
                    standard_deviation_list_train.append(np.sqrt(sum_train / cnt_train))

            if cnt_test >= 1:
                for _ in range(cnt_test):
                    σ_pred_standard_deviation_list_test.append(i)
                    standard_deviation_list_test.append(np.sqrt(sum_test / cnt_test))

        # 横軸を|y-θ|、縦軸をσにした場合のグラフを計算
        standard_deviation_list_test_v2: list[float] = []
        σ_pred_standard_deviation_list_test_v2: list[float] = []
        for i in range(self.human_property.max_age):
            test_list: list[float] = []
            sum_test: float = 0.0
            cnt_test: int = 0

            for j, σ in enumerate(σ_true_list_test):
                if int(σ) == i:
                    test_list.append(σ_pred_list_test[j])
                    sum_test += σ_pred_list_test[j] ** 2
                    cnt_test += 1

            if cnt_test > 1:
                for _ in range(cnt_test):
                    standard_deviation_list_test_v2.append(i)
                    σ_pred_standard_deviation_list_test_v2.append(np.sqrt(sum_test / cnt_test))

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_standard_deviation_list_train, standard_deviation_list_train,
                   bins=self.human_property.max_age,
                   range=[(0, self.human_property.max_age), (0, self.human_property.max_age)], cmin=1)
        plt.colorbar()
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel("RMSE")
        plt.xlim(0, self.human_property.max_age)
        plt.ylim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_standard_deviation_train.png")

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_standard_deviation_list_test, standard_deviation_list_test,
                   bins=self.human_property.max_age,
                   range=[(0, self.human_property.max_age), (0, self.human_property.max_age)], cmin=1)
        plt.colorbar()
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel("RMSE")
        plt.xlim(0, self.human_property.max_age)
        plt.ylim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_standard_deviation_test.png")

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(standard_deviation_list_train, σ_pred_standard_deviation_list_train,
                   bins=self.human_property.max_age,
                   range=[(0, self.human_property.max_age), (0, self.human_property.max_age)], cmin=1)
        plt.colorbar()
        plt.xlabel("RMSE")
        plt.ylabel(r"$\hat{\sigma}$")
        plt.xlim(0, self.human_property.max_age)
        plt.ylim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/standard_deviation_σ_train.png")

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(standard_deviation_list_test_v2, σ_pred_standard_deviation_list_test_v2,
                   bins=self.human_property.max_age,
                   range=[(0, self.human_property.max_age), (0, self.human_property.max_age)], cmin=1)
        plt.colorbar()
        plt.xlabel("RMSE")
        plt.ylabel(r"$\hat{\sigma}$")
        plt.xlim(0, self.human_property.max_age)
        plt.ylim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/standard_deviation_σ_test.png")

        plt.figure()
        plt.hist(σ_pred_standard_deviation_list_train, bins=self.human_property.max_age)
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel("samples")
        plt.xlim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_count_train.png")

        plt.figure()
        plt.hist(σ_pred_standard_deviation_list_test, bins=self.human_property.max_age)
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel("samples")
        plt.xlim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_count_test.png")

        return

        plt.figure()
        plt.hist(standard_deviation_list_train, bins=self.human_property.max_age)
        plt.xlabel(r"$|x - \hat{x}|$")
        plt.ylabel("samples")
        plt.xlim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/residual_error_count_train.png")

        plt.figure()
        plt.hist(standard_deviation_list_test_v2, bins=self.human_property.max_age)
        plt.xlabel(r"$|x - \hat{x}|$")
        plt.ylabel("samples")
        plt.xlim(0, self.human_property.max_age)
        plt.savefig(f"{self.path_property.heatmap_path}/residual_error_count_test.png")

        # 残差標準偏差のエラーバー（標本平均値の標準偏差の標準偏差）を作成
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.errorbar(σ_error_bar_x_list_train, standard_deviation_list_train, yerr=σ_error_bar_list_train, capsize=5,
                     fmt='o', markersize=5, ecolor='black', markeredgecolor="black", color='w')
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel("RMSE")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_error_bar_train.png")

        plt.figure()
        plt.axes().set_aspect('equal')
        plt.errorbar(σ_error_bar_x_list_test, standard_deviation_list_test, yerr=σ_error_bar_list_test, capsize=5,
                     fmt='o', markersize=5, ecolor='black', markeredgecolor="black", color='w')
        plt.xlabel(r"$\hat{\sigma}$")
        plt.ylabel("RMSE")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(f"{self.path_property.heatmap_path}/σ_error_bar_test.png")
